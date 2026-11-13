#!/bin/bash
# ff10build.sh [DIR] -- FFmpeg for the rad10 media path, as radiance's Dockerfile `ffmpeg` stage (8.1.3, sha256-checked;
# its demuxer / decoder / parser lists) minus what box2 / ds8 lack and we do not install: no libdav1d (AV1 input refused),
# no libwebp encoder, no libopenh264 (generated video), no nasm (--disable-x86asm). Images (png, jpeg, webp, bmp, gif,
# tiff, ...), video and audio decode as in the image. Shared, into DIR/inst (default /data/q27/rad10/ffmpeg), pkgconf from
# pip in DIR/venv. CPU only, niced. Idempotent. Log DIR/ffbuild.log. (w130)
set -e; D=${1:-/data/q27/rad10/ffmpeg}; V=8.1.3; SHA=7138d28c96d9d3e3af4ee3d8cad72741f8ffb40da90c1112235dea3ecd3178a3
mkdir -p $D; exec >> $D/ffbuild.log 2>&1; cd $D; echo "$(date -u +%T) start"
[ -x venv/bin/pkgconf ] || { python3 -m venv venv && venv/bin/pip install -q pkgconf; }
if [ ! -f inst/lib/libavformat.so ]; then
  [ -f ffmpeg-$V.tar.xz ] || curl -fsSL --retry 3 -o ffmpeg-$V.tar.xz https://ffmpeg.org/releases/ffmpeg-$V.tar.xz
  echo "$SHA  ffmpeg-$V.tar.xz" | sha256sum -c -
  rm -rf ffmpeg-$V && tar xf ffmpeg-$V.tar.xz && cd ffmpeg-$V
  PKG_CONFIG=$D/venv/bin/pkgconf ./configure --prefix=$D/inst --enable-shared --disable-static --disable-debug \
    --disable-programs --disable-doc --disable-network --disable-autodetect --disable-everything --disable-x86asm \
    --disable-avdevice --disable-avfilter --enable-swresample --enable-zlib \
    --enable-demuxer=mov,matroska,avi,mpegts,flv,ogg,gif,apng,image2pipe,image_png_pipe,image_jpeg_pipe,image_jpegls_pipe,image_jpegxl_pipe,image_webp_pipe,image_bmp_pipe,image_tiff_pipe,image_qoi_pipe,image_pgm_pipe,image_ppm_pipe,image_pbm_pipe,image_pam_pipe,image_gif_pipe,wav,w64,mp3,flac,aac,aiff \
    --enable-decoder=h264,hevc,vp8,vp9,mpeg4,mpeg1video,mpeg2video,mjpeg,jpegls,png,apng,gif,webp,bmp,tiff,qoi,pgm,ppm,pbm,pam,pgmyuv,theora,vp3,flv,h263,pcm_u8,pcm_s16le,pcm_s16be,pcm_s24le,pcm_s24be,pcm_s32le,pcm_s32be,pcm_f32le,pcm_f32be,pcm_f64le,pcm_alaw,pcm_mulaw,mp3float,mp2float,flac,vorbis,opus,aac \
    --enable-encoder=png,mjpeg,aac --enable-muxer=mp4 \
    --enable-parser=h264,hevc,vp8,vp9,av1,mpeg4video,mpegvideo,mjpeg,png,gif,bmp,webp,vp3,qoi,pnm,h263,jpegxl,mpegaudio,flac,aac,vorbis,opus
  nice -n 15 make -j ${J:-16} && make install
  cd $D
fi
PKG_CONFIG_PATH=$D/inst/lib/pkgconfig venv/bin/pkgconf --modversion libavformat libavcodec libavutil libswscale libswresample
echo "$(date -u +%T) FF READY $D/inst"
