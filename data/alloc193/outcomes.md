| allocator | files | stale (slice rotated) | stage-1 decided | tentative | confirmed accept | confirmed join | stage-2 reject | stage-2 pending | confirmed at (A accept, j join) |
|---|---|---|---|---|---|---|---|---|---|
| anom | 7 | 0 | 7 | 7 | 1 | 2 | 4 | 0 | 4.5j 2.5j 2A |
| cons | 5 | 0 | 5 | 2 | 0 | 1 | 1 | 0 | 4.5j |
| tr | 4 | 0 | 4 | 2 | 1 | 0 | 1 | 0 | 2.5A |
| recal | 4 | 0 | 4 | 3 | 0 | 1 | 2 | 0 | 3j |
| ens | 21 | 13 | 8 | 4 | 1 | 0 | 1 | 2 | 2.5A |
| curve | 5 | 0 | 5 | 1 | 0 | 0 | 1 | 0 |  |
| rob | 3 | 0 | 3 | 0 | 0 | 0 | 0 | 0 |  |
| rankreg | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |

| file | alloc | size | tag | stage 1 (log r, z, docs) | stage 2 (log r, z) |
|---|---|---|---|---|---|
| w172-cons-4.json | cons | 4 | manual-01556 | reject +0.0107 z +0.59 k 162 stopped | - |
| w172-cons-2.5.json | cons | 2.5 | manual-01558 | tentative -0.0306 z -4.92 k 299 | reject +0.0008 z +0.07 |
| w172-cons-3.5.json | cons | 3.5 | manual-01560 | reject +0.0154 z +0.56 k 162 stopped | - |
| w172-cons-4.5.json | cons | 4.5 | manual-01563 | tentative -0.0107 z -0.84 k 299 | join -0.0122 z -1.22 |
| w172-cons-5.json | cons | 5 | manual-01564 | reject -0.0003 z -0.03 k 299 | - |
| w172-anom-ffn_downd7-4.5.json | anom | 4.5 | manual-01580 | tentative -0.0038 z -0.49 k 299 | join -0.0109 z -2.35 |
| w172-anom-ffn_gate_upd0-2.5.json | anom | 2.5 | manual-01581 | tentative -0.0064 z -2.05 k 299 | reject +0.0042 z +0.92 |
| w172-curve-2.5.json | curve | 2.5 | manual-01589 | tentative -0.0079 z -0.73 k 299 | reject -0.0012 z -0.12 |
| w172-anom-ffn_downd0-3.json | anom | 3 | manual-01590 | tentative -0.0053 z -1.63 k 299 | reject +0.0005 z +0.17 |
| w172-tr-3.json | tr | 3 | manual-01591 | reject +0.0029 z +0.26 k 233 | - |
| w172-bootp-3.json | bootp | 3 | probe-01592 | probe KLD 0.02280, log r vs incumbent +0.0081 | - |
| w172-curve-2.json | curve | 2 | manual-01608 | reject +0.0876 z +6.18 k 31 stopped | - |
| w172-anom-ffn_downd6-2.5.json | anom | 2.5 | manual-01609 | tentative -0.0206 z -2.25 k 299 | join -0.0184 z -1.93 |
| w172-recal-4.json | recal | 4 | manual-01614 | tentative -0.0170 z -0.87 k 299 | reject +0.0091 z +0.58 |
| w172-curve-4.json | curve | 4 | manual-01618 | reject +0.0097 z +0.59 k 162 | - |
| w172-bootp-4.json | bootp | 4 | probe-01619 | probe KLD 0.00627, log r vs incumbent +0.0082 | - |
| w172-anom-ffn_downd7-6.json | anom | 6 | manual-01624 | tentative -0.0155 z -1.54 k 299 | reject -0.0070 z -0.69 |
| w172-tr-2.5.json | tr | 2.5 | manual-01628 | tentative -0.0071 z -1.10 k 299 | accept -0.0194 z -2.38 |
| w172-rob-3.json | rob | 3 | manual-01635 | reject +0.0134 z +1.32 k 162 stopped | - |
| w172-curve-3.5.json | curve | 3.5 | manual-01641 | reject +0.0079 z +0.53 k 164 stopped | - |
| w172-anom-ffn_gate_upd0-2.json | anom | 2 | manual-01645 | tentative -0.0040 z -1.47 k 299 | accept -0.0055 z -1.64 |
| w172-recal-3.json | recal | 3 | manual-01648 | tentative -0.0242 z -2.98 k 299 | join -0.0103 z -1.21 |
| w172-bootm-3.json | bootm | 3 | probe-01652 | probe KLD 0.02262, log r vs incumbent +0.0002 | - |
| w172-curve-4.5.json | curve | 4.5 | manual-01658 | reject +0.0498 z +1.86 k 47 stopped | - |
| w172-anom-ffn_downd6-3.json | anom | 3 | manual-01662 | tentative -0.0015 z -0.53 k 299 | reject +0.0050 z +1.71 |
| w172-tr-4.json | tr | 4 | manual-01663 | tentative -0.0051 z -0.47 k 299 | reject +0.0064 z +0.40 |
| w172-bootp-3.5.json | bootp | 3.5 | probe-01674 | probe KLD 0.01182, log r vs incumbent -0.0012 | - |
| w172-recal-2.5.json | recal | 2.5 | manual-01676 | tentative -0.0059 z -0.77 k 299 | reject +0.0225 z +2.34 |
| w172-tr-5.json | tr | 5 | manual-01681 | reject +0.0047 z +0.32 k 201 | - |
| w172-rob-4.json | rob | 4 | manual-01683 | reject +0.0131 z +0.69 k 162 | - |
| w172-recal-5.json | recal | 5 | manual-01684 | reject +0.0006 z +0.05 k 274 stopped | - |
| w172-rob-5.json | rob | 5 | manual-01688 | reject +0.0016 z +0.16 k 299 | - |
| w172-bootm-4.json | bootm | 4 | probe-01689 | probe KLD 0.00624, log r vs incumbent +0.0027 | - |
| w172-ens-2-w0.5-R0.25.json | ens | 2 | manual-01691 | tentative -0.0038 z -0.80 k 299 | pending |
| w172-curve-3.125.json | curve | 3.125 | probe-01694 | probe pending | - |
| w172-ens-2.5-w0.5-R0.25.json | ens | 2.5 | manual-01695 | tentative -0.0188 z -3.13 k 299 | accept -0.0104 z -1.44 |
| w172-curve-4.375.json | curve | 4.375 | probe-01696 | probe pending | - |
| w172-ens-3-w0.5-R0.25.json | ens | 3 | manual-01699 | reject +0.0016 z +0.15 k 226 stopped | - |
| w172-ens-3.5-w0.5-R0.25.json | ens | 3.5 | manual-01700 | tentative -0.0184 z -1.70 k 299 | reject +0.0137 z +0.71 |
| w172-ens-4-w0.5-R0.25.json | ens | 4 | manual-01701 | tentative -0.0176 z -1.07 k 299 | pending |
| w172-ens-4.5-w0.5-R0.25.json | ens | 4.5 | manual-01706 | reject +0.0754 z +1.91 k 54 stopped | - |
| w172-ens-5-w0.5-R0.25.json | ens | 5 | manual-01707 | reject +0.0043 z +0.32 k 201 stopped | - |
| w172-ens-6-w0.5-R0.25.json | ens | 6 | manual-01708 | reject +0.0180 z +0.92 k 299 | - |
| w172-ens-2.5-w0.5-R0.5.json | ens | 2.5 | manual-01710 | stale | - |
| w172-ens-3-w0.5-R0.5.json | ens | 3 | manual-01712 | stale | - |
| w172-ens-3.5-w0.5-R0.5.json | ens | 3.5 | manual-01716 | stale | - |
| w172-ens-4-w0.5-R0.5.json | ens | 4 | manual-01717 | stale | - |
| w172-ens-5-w0.5-R0.5.json | ens | 5 | manual-01720 | stale | - |
| w172-ens-6-w0.5-R0.5.json | ens | 6 | manual-01721 | stale | - |
| w172-ens-2-w0.75-R0.25.json | ens | 2 | manual-01722 | stale | - |
| w172-ens-2.5-w0.75-R0.25.json | ens | 2.5 | manual-01723 | stale | - |
| w172-ens-3-w0.75-R0.25.json | ens | 3 | manual-01726 | stale | - |
| w172-ens-3.5-w0.75-R0.25.json | ens | 3.5 | manual-01728 | stale | - |
| w172-ens-4-w0.75-R0.25.json | ens | 4 | manual-01729 | stale | - |
| w172-ens-4.5-w0.75-R0.25.json | ens | 4.5 | manual-01730 | stale | - |
| w172-ens-5-w0.75-R0.25.json | ens | 5 | manual-01731 | stale | - |
| w172-rankreg-3-1.json | rankreg | 3 | manual-01765 | pending | - |
| w172-rankreg-3.5-1.json | rankreg | 3.5 | manual-01780 | pending | - |
