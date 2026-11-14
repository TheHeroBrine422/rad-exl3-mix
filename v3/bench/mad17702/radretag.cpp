/* radretag -- a .rad copied with its exl3i encodings' codebook renamed: `radretag IN.rad OUT.rad FROM TO` rewrites every
 * encoding whose scheme is exl3i and whose transform is "FROM/h128/..." to "TO/h128/..." (the 24-byte transform field of
 * the encoding table, RadEncoding; nothing else changes: the planes, the directory, the strings). For #177-02 (w150):
 * w139's stage-3 hash levels were packed with their trellis labelled mul1i8 (the EXL3I_BENCH_IQRH build read them); the
 * codes are mad's exactly, so `radretag mix-a8.rad mix-a8-mad.rad mul1i8 mad` gives the container exl3i's codebook mad
 * reads. The copy is a plain byte copy (copy_file_range); the result is checked to differ only in those fields.
 * Build: g++ -O2 -I<radiance>/abi radretag.cpp -o radretag */
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include "rad_format.h"
#include "rad_encoding.h"

int main(int argc, char** argv) {
    if (argc != 5) { std::fprintf(stderr, "radretag IN.rad OUT.rad FROM TO\n"); return 2; }
    const std::string from = std::string(argv[3]) + "/", to = std::string(argv[4]) + "/";
    int in = open(argv[1], O_RDONLY);
    if (in < 0) { perror(argv[1]); return 1; }
    struct stat st;
    fstat(in, &st);
    const std::string part = std::string(argv[2]) + ".part";
    int out = open(part.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (out < 0) { perror(part.c_str()); return 1; }
    for (off_t left = st.st_size; left > 0;) {
        const ssize_t n = copy_file_range(in, nullptr, out, nullptr, (size_t)left, 0);
        if (n <= 0) { perror("copy_file_range"); return 1; }
        left -= n;
    }
    uint8_t* m = (uint8_t*)mmap(nullptr, (size_t)st.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, out, 0);
    if (m == MAP_FAILED) { perror("mmap"); return 1; }
    const RadFileHeader* h = (const RadFileHeader*)m;
    if (h->magic != RAD_MAGIC || h->version != RAD_FORMAT_VER) { std::fprintf(stderr, "%s: not RAD2\n", argv[1]); return 1; }
    RadEncoding* e = (RadEncoding*)(m + h->enc_off);
    int n = 0, other = 0;
    for (uint64_t i = 0; i < h->enc_count; ++i) {
        if (std::strncmp(e[i].scheme, "exl3i", RAD_ENC_STR)) continue;
        const std::string t(e[i].transform, strnlen(e[i].transform, RAD_ENC_STR));
        if (t.compare(0, from.size(), from)) { ++other; continue; }
        const std::string nt = to + t.substr(from.size());
        if (nt.size() >= RAD_ENC_STR) { std::fprintf(stderr, "%s: too long\n", nt.c_str()); return 1; }
        rad_enc_copy_str(e[i].transform, nt.c_str());
        ++n;
    }
    msync(m, (size_t)st.st_size, MS_SYNC);
    munmap(m, (size_t)st.st_size);
    close(out);
    close(in);
    if (rename(part.c_str(), argv[2])) { perror("rename"); return 1; }
    std::printf("%s: %d exl3i encodings %s -> %s (%d other exl3i left as they were), %llu encodings in all\n", argv[2], n,
                argv[3], argv[4], other, (unsigned long long)h->enc_count);
    return 0;
}
