P="python3 -I /workspace/rad-exl3-mix/proxy2b/pcmp.py /data/g08-search1"; Y=/data/yb; D=/data/yb/d1103; W=/data/yb/w11-review
row() { r=$($P $2 $3 2>/dev/null) || { echo "$1: missing"; return; }; python3 -c "
import json,sys; r=json.loads(sys.argv[1]); print(f\"{sys.argv[2]:34s} mean {r['ratio']:.3f}  KL99 {r['kl99_ratio']:.3f} [{r['kl99_ci'][0]:.3f},{r['kl99_ci'][1]:.3f}]  KL99.9 {r['kl999_ratio']:.3f} [{r['kl999_ci'][0]:.3f},{r['kl999_ci'][1]:.3f}]  top1 {100*r['top1_diff']:+.2f}pt\")" "$r" "$1"; }
cd /tmp
for k in 3 2; do
row "K$k H_out (edy vs base)" $Y/k${k}base.f32 $Y/k${k}edy.f32
row "K$k gen-mask Σdy² (dygm vs edy)" $Y/k${k}edy.f32 $D/k${k}dygm.f32
row "K$k #87 asym ρ.5 vs base" $Y/k${k}base.f32 $Y/w07-seqsib/k${k}asym.f32
row "K$k #87 on edy (ρ.5)" $Y/k${k}edy.f32 $D/k${k}edyasym.f32
row "K$k #87 ρ0 on edy vs ρ.5" $D/k${k}edyasym.f32 $D/k${k}edyasym_r0.f32
row "K$k healing on edy+asym (norms)" $D/k${k}edyasym.f32 $W/k${k}edyasym_healn.f32
row "K$k C10 swj on base" $Y/k${k}base.f32 $Y/w02-fixedscale/k${k}base_swj.f32
row "K$k i8c on base (cub)" $Y/k${k}base.f32 $Y/k${k}cub.f32
done
