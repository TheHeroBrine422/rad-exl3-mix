/* radmix -- a .rad container assembled from others, weight by weight, by reference (FICLONERANGE: on a reflink
 * filesystem every weight is a block-sharing clone, so a mix costs no copy and no disk).
 *
 *   radmix OUT.rad BASE.rad MIXFILE
 *   MIXFILE lines: <weight name> <donor.rad>     ('#' comments; weights not named come from BASE)
 *                  act <weight name> <mode>      the weight's activation mode, metadata key exl3.act.<weight name>
 *                                                (exl3_act.h: f16 | a8t | a8g<N>, N a multiple of 128). The mix's
 *                                                metadata is BASE's with these keys set; with no act line BASE's
 *                                                metadata table is kept as is.
 *
 * The donor must hold a weight of the same name and logical shape; its encoding, planes, quantiser and options
 * come along. BASE's whole pre-data region (header, strings, metadata, the old tables, vocab, profile) is kept
 * byte for byte, so every absolute offset in it stays valid; the new string blob (BASE's + the donors' strings),
 * directory, plane table and encoding table are appended after it and the header points at them. Weights keep
 * BASE's order, each RAD_ALIGN_UNIT-aligned, from a RAD_ALIGN_POOL-aligned data section. Falls back to a byte copy
 * where a clone is refused. */
#include <fcntl.h>
#include <linux/fs.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include "rad_format.h"
#include "exl3_act.h"

struct Src {
    int fd; size_t size; const uint8_t* m; const RadFileHeader* h;
    const char* str() const { return (const char*)m + h->str_off; }
    const RadFileEntry* dir() const { return (const RadFileEntry*)(m + h->dir_off); }
    const RadFilePlane* pl() const { return (const RadFilePlane*)(m + h->plane_off); }
    const RadEncoding* enc() const { return (const RadEncoding*)(m + h->enc_off); }
    std::map<std::string, uint64_t> idx;
};
static Src* open_src(const std::string& p) {
    static std::map<std::string, Src*> cache;
    auto it = cache.find(p); if (it != cache.end()) return it->second;
    Src* s = new Src{};
    s->fd = open(p.c_str(), O_RDONLY);
    if (s->fd < 0) { perror(p.c_str()); exit(1); }
    struct stat st; fstat(s->fd, &st); s->size = st.st_size;
    s->m = (const uint8_t*)mmap(nullptr, s->size, PROT_READ, MAP_SHARED, s->fd, 0);
    s->h = (const RadFileHeader*)s->m;
    if (s->h->magic != RAD_MAGIC || s->h->version != RAD_FORMAT_VER) { fprintf(stderr, "%s: not RAD2\n", p.c_str()); exit(1); }
    for (uint64_t i = 0; i < s->h->dir_count; ++i) s->idx[s->str() + s->dir()[i].name] = i;
    return cache[p] = s;
}
static uint64_t al(uint64_t x, uint64_t a) { return (x + a - 1) / a * a; }

static void put(int fd, uint64_t off, const void* p, size_t n) {
    const uint8_t* b = (const uint8_t*)p;
    while (n) { ssize_t w = pwrite(fd, b, n, off); if (w <= 0) { perror("pwrite"); exit(1); } b += w; off += w; n -= w; }
}
static int n_clone = 0, n_copy = 0;
static void move(int dfd, uint64_t doff, const Src* s, uint64_t soff, uint64_t n) {
    struct file_clone_range r{}; r.src_fd = s->fd; r.src_offset = soff; r.dest_offset = doff;
    r.src_length = al(n, RAD_ALIGN_UNIT);
    if (soff + r.src_length > s->size) r.src_length = 0;          /* to the source's end: 0 = "to EOF" */
    if (soff % RAD_ALIGN_UNIT == 0 && doff % RAD_ALIGN_UNIT == 0 && ioctl(dfd, FICLONERANGE, &r) == 0) { ++n_clone; return; }
    put(dfd, doff, s->m + soff, n); ++n_copy;
}

int main(int argc, char** argv) {
    if (argc != 4) { fprintf(stderr, "radmix OUT.rad BASE.rad MIXFILE\n"); return 2; }
    Src* B = open_src(argv[2]);
    std::map<std::string, std::string> mix, act;
    { FILE* f = fopen(argv[3], "r"); if (!f) { perror(argv[3]); return 1; }
      char a[512], b[1024], c[64], line[2048];
      while (fgets(line, sizeof line, f)) {
          if (line[0] == '#') continue;
          if (sscanf(line, "act %1023s %63s", b, c) == 2) { act[b] = c; continue; }
          if (sscanf(line, "%511s %1023s", a, b) != 2) continue;
          mix[a] = b;
      }
      fclose(f); }
    for (auto& kv : mix) if (!B->idx.count(kv.first)) { fprintf(stderr, "base has no weight %s\n", kv.first.c_str()); return 1; }
    for (auto& kv : act) {
        if (!B->idx.count(kv.first)) { fprintf(stderr, "act: base has no weight %s\n", kv.first.c_str()); return 1; }
        if (exl3_act_parse(kv.second.c_str(), nullptr) != 0) {
            fprintf(stderr, "act %s %s: not an activation mode (f16, a8t, a8g<N>)\n", kv.first.c_str(), kv.second.c_str());
            return 1;
        }
    }

    const RadFileHeader& bh = *B->h;
    std::string strs(B->str(), bh.str_bytes);
    auto intern = [&](const char* s) -> rad_stroff {
        if (!*s) return 0;
        size_t p = strs.find(std::string(s) + '\0');
        if (p != std::string::npos && (p == 0 || strs[p - 1] == 0)) return p;
        p = strs.size(); strs += s; strs += '\0'; return p;
    };
    std::vector<RadEncoding> encs(B->enc(), B->enc() + bh.enc_count);
    auto enc_of = [&](const RadEncoding& e) -> uint32_t {
        for (size_t i = 0; i < encs.size(); ++i) if (!memcmp(&encs[i], &e, sizeof e)) return (uint32_t)i;
        encs.push_back(e); return (uint32_t)encs.size() - 1;
    };

    /* the directory: each entry from its source, offsets assigned below */
    struct Pick { const Src* s; const RadFileEntry* e; };
    std::vector<Pick> picks; std::vector<RadFileEntry> dir; std::vector<RadFilePlane> planes;
    for (uint64_t i = 0; i < bh.dir_count; ++i) {
        const RadFileEntry& be = B->dir()[i];
        std::string name = B->str() + be.name;
        const Src* s = B; const RadFileEntry* e = &be;
        auto m = mix.find(name);
        if (m != mix.end()) {
            s = open_src(m->second);
            auto j = s->idx.find(name);
            if (j == s->idx.end()) { fprintf(stderr, "%s has no weight %s\n", m->second.c_str(), name.c_str()); return 1; }
            e = &s->dir()[j->second];
            if (e->rank != be.rank || memcmp(e->shape, be.shape, sizeof be.shape)) { fprintf(stderr, "%s: shape differs in %s\n", name.c_str(), m->second.c_str()); return 1; }
        }
        RadFileEntry d = *e;
        d.name = be.name;
        d.quantizer = intern(s->str() + e->quantizer);
        d.options = intern(s->str() + e->options);
        d.enc = enc_of(s->enc()[e->enc]);
        d.plane_first = planes.size();
        for (uint32_t p = 0; p < e->n_planes; ++p) planes.push_back(s->pl()[e->plane_first + p]);  /* offsets fixed below */
        dir.push_back(d); picks.push_back({s, e});
    }

    /* the metadata: BASE's (its string offsets are valid in the new blob, which starts with BASE's), the act keys
     * set; appended as a new table only when there are act lines */
    std::vector<RadFileKV> meta((const RadFileKV*)(B->m + bh.meta_off), (const RadFileKV*)(B->m + bh.meta_off) + bh.meta_count);
    for (auto& kv : act) {
        const std::string key = EXL3_ACT_KEY_PREFIX + kv.first;
        RadFileKV v{}; v.key = intern(key.c_str()); v.type = RAD_P_STR; v.v.s = intern(kv.second.c_str());
        bool set = false;
        for (auto& m : meta) if (!strcmp(strs.c_str() + m.key, key.c_str())) { m = v; set = true; }
        if (!set) meta.push_back(v);
    }

    /* layout: BASE's pre-data region, then the new tables, then the data */
    uint64_t pre = bh.data_off;   /* everything before BASE's data */
    uint64_t o = al(pre, 256);
    RadFileHeader h = bh;
    h.str_off = o; h.str_bytes = strs.size();                     o = al(o + strs.size(), 256);
    h.dir_off = o; h.dir_count = dir.size();                      o = al(o + dir.size() * sizeof(RadFileEntry), 256);
    h.plane_off = o; h.plane_count = planes.size();               o = al(o + planes.size() * sizeof(RadFilePlane), 256);
    h.enc_off = o; h.enc_count = encs.size();                     o = al(o + encs.size() * sizeof(RadEncoding), 256);
    if (!act.empty()) { h.meta_off = o; h.meta_count = meta.size(); o = al(o + meta.size() * sizeof(RadFileKV), 256); }
    h.data_off = al(o, RAD_ALIGN_POOL);
    uint64_t cur = h.data_off;
    for (size_t i = 0; i < dir.size(); ++i) {
        RadFileEntry& d = dir[i]; const RadFileEntry* e = picks[i].e;
        cur = al(cur, RAD_ALIGN_UNIT);
        for (uint32_t p = 0; p < d.n_planes; ++p) planes[d.plane_first + p].offset = cur + (planes[d.plane_first + p].offset - e->offset);
        d.offset = cur; cur += d.bytes;
    }
    h.data_bytes = al(cur, RAD_ALIGN_UNIT) - h.data_off;
    h.file_bytes = h.data_off + h.data_bytes;

    int fd = open(argv[1], O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) { perror(argv[1]); return 1; }
    if (ftruncate(fd, h.file_bytes)) { perror("ftruncate"); return 1; }
    put(fd, 0, B->m, pre);
    put(fd, h.str_off, strs.data(), strs.size());
    put(fd, h.dir_off, dir.data(), dir.size() * sizeof(RadFileEntry));
    put(fd, h.plane_off, planes.data(), planes.size() * sizeof(RadFilePlane));
    put(fd, h.enc_off, encs.data(), encs.size() * sizeof(RadEncoding));
    if (!act.empty()) put(fd, h.meta_off, meta.data(), meta.size() * sizeof(RadFileKV));
    for (size_t i = 0; i < dir.size(); ++i) move(fd, dir[i].offset, picks[i].s, picks[i].e->offset, dir[i].bytes);
    put(fd, 0, &h, sizeof h);
    if (ftruncate(fd, h.file_bytes)) { perror("ftruncate"); return 1; }
    close(fd);
    fprintf(stderr, "%s: %zu weights (%zu from donors, %zu act keys), %.3f GiB, %d cloned, %d copied\n", argv[1], dir.size(),
            mix.size(), act.size(), h.file_bytes / 1073741824.0, n_clone, n_copy);
    return 0;
}
