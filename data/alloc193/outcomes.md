| allocator | files | stale (slice rotated) | stage-1 decided | tentative | confirmed accept | confirmed join | stage-2 reject | stage-2 pending | confirmed at (A accept, j join) |
|---|---|---|---|---|---|---|---|---|---|
| anom | 41 | 0 | 40 | 24 | 2 | 5 | 14 | 3 | 4.5j 2.5j 2A 4A 5j 2.5j 2j |
| rankreg | 56 | 0 | 56 | 18 | 1 | 5 | 12 | 0 | 2.5j 2A 2j 6j 2j 4.5j |
| ens | 76 | 13 | 59 | 22 | 3 | 1 | 18 | 0 | 2.5A 2.5A 3.5A 5j |
| xplant | 41 | 0 | 28 | 11 | 2 | 1 | 8 | 0 | 3.5A 2.5j 2A |
| xavg | 16 | 0 | 16 | 9 | 2 | 0 | 7 | 0 | 2.5A 5A |
| cons | 5 | 0 | 5 | 2 | 0 | 1 | 1 | 0 | 4.5j |
| tr | 18 | 0 | 18 | 7 | 1 | 0 | 6 | 0 | 2.5A |
| recal | 12 | 0 | 12 | 3 | 0 | 1 | 2 | 0 | 3j |
| curve | 5 | 0 | 5 | 1 | 0 | 0 | 1 | 0 |  |
| rob | 3 | 0 | 3 | 0 | 0 | 0 | 0 | 0 |  |
| spread | 8 | 0 | 8 | 3 | 0 | 0 | 3 | 0 |  |
| cap | 8 | 0 | 8 | 2 | 0 | 0 | 2 | 0 |  |
| pavg | 6 | 0 | 6 | 3 | 0 | 0 | 3 | 0 |  |
| dxp | 11 | 0 | 11 | 0 | 0 | 0 | 0 | 0 |  |
| mom | 8 | 0 | 8 | 0 | 0 | 0 | 0 | 0 |  |
| floor | 6 | 0 | 2 | 0 | 0 | 0 | 0 | 0 |  |

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
| w172-ens-2-w0.5-R0.25.json | ens | 2 | manual-01691 | tentative -0.0038 z -0.80 k 299 | reject +0.0060 z +0.94 |
| w172-curve-3.125.json | curve | 3.125 | probe-01694 | probe pending | - |
| w172-ens-2.5-w0.5-R0.25.json | ens | 2.5 | manual-01695 | tentative -0.0188 z -3.13 k 299 | accept -0.0104 z -1.44 |
| w172-curve-4.375.json | curve | 4.375 | probe-01696 | probe pending | - |
| w172-ens-3-w0.5-R0.25.json | ens | 3 | manual-01699 | reject +0.0016 z +0.15 k 226 stopped | - |
| w172-ens-3.5-w0.5-R0.25.json | ens | 3.5 | manual-01700 | tentative -0.0184 z -1.70 k 299 | reject +0.0137 z +0.71 |
| w172-ens-4-w0.5-R0.25.json | ens | 4 | manual-01701 | tentative -0.0176 z -1.07 k 299 | reject +0.0114 z +0.79 |
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
| w172-rankreg-3-1.json | rankreg | 3 | manual-01765 | tentative -0.0158 z -1.86 k 313 | reject +0.0114 z +1.36 |
| w172-rankreg-3.5-1.json | rankreg | 3.5 | manual-01780 | reject +0.0079 z +1.04 k 110 | - |
| w172-rankreg-4-1.json | rankreg | 4 | manual-01784 | reject +0.1098 z +2.81 k 17 | - |
| w172-rankreg-4.5-1.json | rankreg | 4.5 | manual-01785 | reject +0.0099 z +1.25 k 213 stopped | - |
| w172-rankreg-2.5-1.json | rankreg | 2.5 | manual-01794 | tentative -0.0263 z -2.41 k 313 | join -0.0193 z -2.50 |
| w172-rankreg-5-1.json | rankreg | 5 | manual-01795 | tentative -0.0256 z -2.22 k 313 | reject -0.0028 z -0.17 |
| w172-rankreg-2-1.json | rankreg | 2 | manual-01796 | tentative -0.0067 z -0.81 k 313 | accept -0.0172 z -2.94 |
| w172-rankreg-6-1.json | rankreg | 6 | manual-01799 | reject +0.0220 z +1.24 k 313 | - |
| w172-rankreg-3-2.json | rankreg | 3 | manual-01804 | reject +0.0125 z +2.08 k 313 | - |
| w172-rankreg-3.5-2.json | rankreg | 3.5 | manual-01805 | reject +0.0578 z +1.53 k 67 | - |
| w172-rankreg-4-2.json | rankreg | 4 | manual-01806 | reject +0.0170 z +0.76 k 213 stopped | - |
| w172-rankreg-4.5-2.json | rankreg | 4.5 | manual-01812 | tentative -0.0046 z -0.54 k 313 | reject -0.0063 z -0.68 |
| w172-rankreg-2.5-2.json | rankreg | 2.5 | manual-01815 | reject +0.0229 z +3.17 k 313 | - |
| w172-rankreg-5-2.json | rankreg | 5 | manual-01820 | tentative -0.0140 z -0.98 k 313 | reject -0.0027 z -0.18 |
| w172-rankreg-2-2.json | rankreg | 2 | manual-01821 | reject +0.0106 z +1.29 k 109 stopped | - |
| w172-rankreg-6-2.json | rankreg | 6 | manual-01824 | reject +0.0364 z +2.04 k 43 | - |
| w172-rankreg-3-3.json | rankreg | 3 | manual-01825 | tentative -0.0089 z -1.26 k 313 | reject +0.0106 z +1.60 |
| w172-rankreg-3.5-3.json | rankreg | 3.5 | manual-01826 | reject +0.0017 z +0.34 k 213 stopped | - |
| w172-rankreg-4-3.json | rankreg | 4 | manual-01827 | reject +0.0317 z +1.04 k 111 | - |
| w172-rankreg-4.5-3.json | rankreg | 4.5 | manual-01828 | reject +0.0548 z +1.51 k 69 stopped | - |
| w172-rankreg-2.5-3.json | rankreg | 2.5 | manual-01829 | reject +0.0154 z +2.20 k 313 | - |
| w172-rankreg-5-3.json | rankreg | 5 | manual-01830 | reject +0.0203 z +2.36 k 313 | - |
| w172-rankreg-2-3.json | rankreg | 2 | manual-01831 | tentative -0.0159 z -1.97 k 313 | join -0.0114 z -2.00 |
| w172-rankreg-6-3.json | rankreg | 6 | manual-01832 | tentative -0.0184 z -1.15 k 313 | join -0.0207 z -2.71 |
| w172-ens-6-w0.75-R0.25.json | ens | 6 | manual-01833 | reject +0.0200 z +1.12 k 313 | - |
| w172-ens-2.5-w0.75-R0.5.json | ens | 2.5 | manual-01834 | tentative -0.0081 z -1.66 k 313 | accept -0.0093 z -1.66 |
| w172-ens-3.5-w0.75-R0.5.json | ens | 3.5 | manual-01835 | tentative -0.0082 z -0.71 k 313 | accept -0.0187 z -2.19 |
| w172-ens-4-w0.75-R0.5.json | ens | 4 | manual-01836 | reject +0.0438 z +1.40 k 109 stopped | - |
| w172-ens-5-w0.75-R0.5.json | ens | 5 | manual-01837 | tentative -0.0328 z -2.51 k 313 | reject -0.0106 z -0.88 |
| w172-ens-6-w0.75-R0.5.json | ens | 6 | manual-01838 | tentative -0.0026 z -0.21 k 313 | reject +0.0175 z +1.51 |
| w172-ens-2-w0.5-R0.5.json | ens | 2 | manual-01841 | tentative -0.0120 z -1.77 k 313 | reject -0.0027 z -0.62 |
| w172-ens-4.5-w0.5-R0.5.json | ens | 4.5 | manual-01842 | tentative -0.0087 z -0.85 k 313 | reject -0.0036 z -0.34 |
| w172-ens-2-w0.75-R0.5.json | ens | 2 | manual-01843 | reject +0.0349 z +2.11 k 43 stopped | - |
| w172-ens-3-w0.75-R0.5.json | ens | 3 | manual-01844 | reject +0.0040 z +0.79 k 313 | - |
| w172-ens-4.5-w0.75-R0.5.json | ens | 4.5 | manual-01845 | tentative -0.0086 z -0.88 k 313 | reject -0.0006 z -0.07 |
| w172-anom-lateffndown-2.json | anom | 2 | manual-01846 | tentative -0.0074 z -2.17 k 313 | reject -0.0017 z -0.49 |
| w172-anom-attn_outputd6-4.5.json | anom | 4.5 | manual-01848 | reject +0.0051 z +0.48 k 313 | - |
| w172-tr-2-R0.125.json | tr | 2 | manual-01849 | reject +0.0077 z +1.02 k 313 | - |
| w172-anom-lateffndown-3.5.json | anom | 3.5 | manual-01851 | reject +0.0015 z +0.43 k 213 stopped | - |
| w172-anom-ffn_gate_upd3-2.5.json | anom | 2.5 | manual-01853 | reject +0.0282 z +2.17 k 45 stopped | - |
| w172-tr-2-R0.25.json | tr | 2 | manual-01854 | tentative -0.0052 z -1.25 k 313 | reject +0.0062 z +1.67 |
| w172-anom-lateffndown-4.5.json | anom | 4.5 | manual-01855 | reject +0.0294 z +1.81 k 54 | - |
| w172-anom-ffn_gate_upd5-4.5.json | anom | 4.5 | manual-01856 | reject +0.0032 z +0.41 k 213 | - |
| w172-tr-2-R0.5.json | tr | 2 | manual-01859 | tentative -0.0058 z -1.36 k 313 | reject +0.0059 z +1.54 |
| w172-anom-lateffndown-4.json | anom | 4 | manual-01872 | tentative -0.0065 z -1.05 k 313 | accept -0.0048 z -1.28 |
| w172-anom-head-3.json | anom | 3 | manual-01873 | tentative -0.0111 z -1.11 k 313 | reject +0.0045 z +0.71 |
| w172-tr-3-R0.125.json | tr | 3 | manual-01874 | reject +0.0132 z +1.71 k 313 | - |
| w172-anom-lateffndown-5.json | anom | 5 | manual-01875 | tentative -0.0212 z -2.82 k 313 | join -0.0099 z -1.19 |
| w172-anom-ssm_outd4-3.json | anom | 3 | manual-01877 | tentative -0.0143 z -1.32 k 313 | reject -0.0032 z -0.50 |
| w172-anom-lateffndown-6.json | anom | 6 | manual-01878 | reject +0.0611 z +1.92 k 48 | - |
| w172-anom-ssm_outd5-2.5.json | anom | 2.5 | manual-01879 | reject +0.0191 z +3.33 k 16 stopped | - |
| w172-xplant-3.5-mid.json | xplant | 3.5 | manual-01880 | tentative -0.0124 z -0.78 k 313 | accept -0.0074 z -1.04 |
| w172-xplant-4-mid.json | xplant | 4 | manual-01882 | reject +0.0533 z +1.62 k 109 stopped | - |
| w172-xplant-5-mid.json | xplant | 5 | manual-01883 | reject +0.0077 z +0.24 k 217 | - |
| w172-xplant-3-mid.json | xplant | 3 | manual-01884 | tentative -0.0128 z -1.28 k 313 | reject +0.0156 z +1.77 |
| w172-xplant-4.5-mid.json | xplant | 4.5 | manual-01885 | reject +0.0522 z +1.24 k 91 stopped | - |
| w172-xplant-2.5-mid.json | xplant | 2.5 | manual-01886 | tentative -0.0096 z -1.28 k 313 | join -0.0244 z -3.67 |
| w172-xplant-6-lo.json | xplant | 6 | manual-01888 | reject +0.0928 z +1.57 k 68 stopped | - |
| w172-xplant-2-hi.json | xplant | 2 | manual-01889 | tentative -0.0110 z -1.95 k 313 | accept -0.0044 z -1.04 |
| w172-spread-3.5-l0.85.json | spread | 3.5 | manual-01890 | reject +0.0025 z +0.22 k 220 stopped | - |
| w172-spread-4-l0.85.json | spread | 4 | manual-01891 | tentative -0.0050 z -0.55 k 313 | reject -0.0100 z -0.88 |
| w172-spread-5-l0.85.json | spread | 5 | manual-01892 | tentative -0.0084 z -0.74 k 313 | reject +0.0005 z +0.03 |
| w172-spread-6-l0.85.json | spread | 6 | manual-01893 | reject +0.0004 z +0.03 k 313 | - |
| w172-spread-3-l0.85.json | spread | 3 | manual-01894 | tentative -0.0145 z -1.24 k 313 | reject +0.0052 z +0.64 |
| w172-spread-4.5-l0.85.json | spread | 4.5 | manual-01895 | reject +0.0488 z +1.18 k 97 stopped | - |
| w172-spread-2.5-l0.85.json | spread | 2.5 | manual-01896 | reject +0.0584 z +3.03 k 16 stopped | - |
| w172-spread-2-l0.85.json | spread | 2 | manual-01897 | reject +0.0239 z +1.09 k 109 stopped | - |
| w172-cap-3.5.json | cap | 3.5 | manual-01903 | tentative -0.0039 z -0.40 k 313 | reject +0.0218 z +2.28 |
| w172-cap-4.json | cap | 4 | manual-01905 | reject -0.0041 z -0.59 k 313 | - |
| w172-cap-5.json | cap | 5 | manual-01907 | reject +0.0176 z +1.08 k 313 | - |
| w172-cap-6.json | cap | 6 | manual-01908 | reject +0.0050 z +0.29 k 313 | - |
| w172-cap-3.json | cap | 3 | manual-01909 | tentative -0.0068 z -0.87 k 313 | reject +0.0078 z +1.01 |
| w172-cap-4.5.json | cap | 4.5 | manual-01911 | reject +0.0256 z +0.94 k 213 stopped | - |
| w172-cap-2.5.json | cap | 2.5 | manual-01913 | reject +0.0159 z +1.99 k 43 stopped | - |
| w172-cap-2.json | cap | 2 | manual-01914 | reject +0.0003 z +0.06 k 265 stopped | - |
| w172-pavg-4.json | pavg | 4 | manual-01916 | tentative -0.0097 z -0.74 k 313 | reject +0.0005 z +0.04 |
| w172-pavg-5.json | pavg | 5 | manual-01917 | reject +0.0502 z +2.38 k 16 stopped | - |
| w172-pavg-3.json | pavg | 3 | manual-01918 | tentative -0.0075 z -1.03 k 313 | reject +0.0018 z +0.21 |
| w172-pavg-4.5.json | pavg | 4.5 | manual-01920 | tentative -0.0126 z -1.29 k 313 | reject -0.0049 z -0.77 |
| w172-pavg-2.5.json | pavg | 2.5 | manual-01921 | reject +0.0094 z +1.32 k 88 | - |
| w172-dxp-2-hi.json | dxp | 2 | manual-01924 | reject +0.0155 z +1.63 k 81 stopped | - |
| w172-ens-2.5-w0.5-R0.5-re.json | ens | 2.5 | manual-01927 | reject +0.0474 z +2.47 k 17 stopped | - |
| w172-tr-2.5R0.125.json | tr | 2.5 | manual-01930 | reject +0.0091 z +1.95 k 43 stopped | - |
| w172-ens-2.5w0.5-R0.25.json | ens | 2.5 | manual-01931 | reject +0.0108 z +1.96 k 45 stopped | - |
| w172-pavg-2.json | pavg | 2 | manual-01932 | reject +0.0038 z +0.83 k 213 | - |
| w172-recal-2.5-i2.json | recal | 2.5 | manual-01933 | reject +0.0435 z +1.99 k 43 stopped | - |
| w172-dxp-2.5-both.json | dxp | 2.5 | manual-01935 | reject +0.0371 z +2.16 k 45 | - |
| w172-ens-3-w0.5-R0.5-re.json | ens | 3 | manual-01936 | reject +0.0070 z +1.09 k 313 | - |
| w172-tr-2.5R0.25.json | tr | 2.5 | manual-01937 | reject +0.0142 z +1.91 k 45 | - |
| w172-ens-2w0.5-R0.25.json | ens | 2 | manual-01938 | tentative -0.0034 z -0.77 k 313 | reject +0.0033 z +0.81 |
| w172-mom-3.5.json | mom | 3.5 | manual-01939 | reject +0.0406 z +1.48 k 81 stopped | - |
| w172-recal-2.json | recal | 2 | manual-01940 | reject +0.0153 z +1.21 k 109 stopped | - |
| w172-dxp-2.5-hi.json | dxp | 2.5 | manual-01941 | reject +0.0266 z +1.59 k 96 stopped | - |
| w172-ens-3.5-w0.5-R0.5-re.json | ens | 3.5 | manual-01942 | tentative -0.0079 z -1.03 k 313 | reject +0.0078 z +1.23 |
| w172-tr-2R0.125.json | tr | 2 | manual-01947 | reject +0.0035 z +0.38 k 213 stopped | - |
| w172-ens-2w0.5-R0.5.json | ens | 2 | manual-01948 | tentative -0.0052 z -1.30 k 313 | reject +0.0052 z +1.33 |
| w172-mom-4.json | mom | 4 | manual-01949 | reject +0.0089 z +0.61 k 213 | - |
| w172-recal-3.5.json | recal | 3.5 | manual-01950 | reject +0.0678 z +1.74 k 69 | - |
| w172-dxp-2.5-lo.json | dxp | 2.5 | manual-01951 | reject +0.0085 z +1.75 k 313 | - |
| w172-ens-4-w0.5-R0.5-re.json | ens | 4 | manual-01952 | reject +0.1375 z +2.84 k 17 stopped | - |
| w172-tr-2R0.25.json | tr | 2 | manual-01953 | tentative -0.0083 z -1.81 k 313 | reject -0.0004 z -0.13 |
| w172-ens-3.5w0.5-R0.25.json | ens | 3.5 | manual-01954 | reject +0.0789 z +1.31 k 86 | - |
| w172-mom-5.json | mom | 5 | manual-01955 | reject +0.0246 z +1.35 k 104 stopped | - |
| w172-recal-3-i2.json | recal | 3 | manual-01956 | reject +0.0255 z +1.24 k 96 stopped | - |
| w172-dxp-3-both.json | dxp | 3 | manual-01957 | reject +0.0008 z +0.07 k 265 stopped | - |
| w172-ens-5-w0.5-R0.5-re.json | ens | 5 | manual-01958 | reject +0.0153 z +1.21 k 109 stopped | - |
| w172-tr-2R0.5.json | tr | 2 | manual-01959 | tentative -0.0057 z -1.39 k 313 | reject +0.0044 z +1.19 |
| w172-ens-3.5w0.5-R0.5.json | ens | 3.5 | manual-01960 | reject +0.0911 z +1.41 k 81 | - |
| w172-mom-6.json | mom | 6 | manual-01961 | reject +0.0116 z +1.21 k 313 | - |
| w172-recal-4.5.json | recal | 4.5 | manual-01962 | reject +0.0028 z +0.18 k 232 | - |
| w172-dxp-3-lo.json | dxp | 3 | manual-01963 | reject +0.0006 z +0.06 k 289 | - |
| w172-rankreg-2-1-i2.json | rankreg | 2 | manual-01964 | tentative -0.0013 z -0.74 k 313 | join -0.0020 z -1.61 |
| w172-ens-6-w0.5-R0.5-re.json | ens | 6 | manual-01965 | tentative -0.0235 z -1.82 k 313 | reject -0.0005 z -0.07 |
| w172-rankreg-2.5-1-i2.json | rankreg | 2.5 | manual-01967 | reject +0.0687 z +2.31 k 16 stopped | - |
| w172-tr-4.5R0.125.json | tr | 4.5 | manual-01968 | reject +0.0006 z +0.05 k 313 | - |
| w172-rankreg-3-1-i2.json | rankreg | 3 | manual-01970 | tentative -0.0161 z -1.60 k 313 | reject +0.0034 z +0.53 |
| w172-ens-4.5w0.5-R0.25.json | ens | 4.5 | manual-01971 | reject +0.0289 z +1.11 k 108 stopped | - |
| w172-rankreg-3.5-1-i2.json | rankreg | 3.5 | manual-01973 | reject +0.1567 z +2.53 k 17 stopped | - |
| w172-mom-3.json | mom | 3 | manual-01974 | reject +0.0267 z +1.36 k 109 stopped | - |
| w172-rankreg-4-1-i2.json | rankreg | 4 | manual-01975 | reject +0.0724 z +2.27 k 16 | - |
| w172-recal-4-i2.json | recal | 4 | manual-01977 | reject +0.0450 z +2.28 k 43 stopped | - |
| w172-rankreg-4.5-1-i2.json | rankreg | 4.5 | manual-01978 | reject +0.0415 z +1.80 k 68 stopped | - |
| w172-dxp-3.5-both.json | dxp | 3.5 | manual-01979 | reject +0.0181 z +0.66 k 213 stopped | - |
| w172-rankreg-5-1-i2.json | rankreg | 5 | manual-01980 | reject +0.0007 z +0.06 k 289 stopped | - |
| w172-ens-2-w0.75-R0.25-re.json | ens | 2 | manual-01982 | reject +0.0309 z +1.76 k 51 | - |
| w172-rankreg-6-1-i2.json | rankreg | 6 | manual-01984 | tentative -0.0096 z -0.82 k 313 | reject +0.0022 z +0.27 |
| w172-tr-4.5R0.25.json | tr | 4.5 | manual-01985 | reject +0.0147 z +1.09 k 110 stopped | - |
| w172-rankreg-2-2-i2.json | rankreg | 2 | manual-01987 | reject +0.0008 z +0.37 k 213 stopped | - |
| w172-ens-4.5w0.5-R0.5.json | ens | 4.5 | manual-01988 | reject +0.0010 z +0.08 k 258 stopped | - |
| w172-rankreg-2.5-2-i2.json | rankreg | 2.5 | manual-01989 | reject +0.0085 z +1.68 k 313 | - |
| w172-mom-4.5.json | mom | 4.5 | manual-01990 | reject +0.0329 z +2.06 k 43 stopped | - |
| w172-rankreg-3-2-i2.json | rankreg | 3 | manual-01991 | reject +0.0137 z +1.49 k 313 | - |
| w172-recal-5-i2.json | recal | 5 | manual-01992 | reject +0.0116 z +0.70 k 213 | - |
| w172-rankreg-3.5-2-i2.json | rankreg | 3.5 | manual-01993 | reject +0.0824 z +1.39 k 88 stopped | - |
| w172-dxp-3.5-hi.json | dxp | 3.5 | manual-01994 | reject +0.0380 z +1.19 k 96 stopped | - |
| w172-rankreg-4-2-i2.json | rankreg | 4 | manual-01995 | tentative -0.0065 z -0.40 k 313 | reject -0.0040 z -0.41 |
| w172-ens-2.5-w0.75-R0.25-re.json | ens | 2.5 | manual-01997 | tentative -0.0119 z -1.48 k 313 | reject -0.0001 z -0.01 |
| w172-rankreg-4.5-2-i2.json | rankreg | 4.5 | manual-02000 | reject +0.0013 z +0.17 k 313 | - |
| w172-tr-4.5R0.5.json | tr | 4.5 | manual-02001 | reject +0.0026 z +0.20 k 232 stopped | - |
| w172-rankreg-5-2-i2.json | rankreg | 5 | manual-02002 | tentative -0.0070 z -0.64 k 313 | reject +0.0021 z +0.14 |
| w172-ens-4w0.5-R0.25.json | ens | 4 | manual-02003 | reject -0.0039 z -0.23 k 313 | - |
| w172-rankreg-6-2-i2.json | rankreg | 6 | manual-02004 | tentative -0.0056 z -0.38 k 313 | reject +0.0146 z +1.31 |
| w172-mom-2.5.json | mom | 2.5 | manual-02005 | reject +0.0874 z +1.84 k 43 stopped | - |
| w172-rankreg-2-3-i2.json | rankreg | 2 | manual-02006 | reject +0.0024 z +1.14 k 112 stopped | - |
| w172-recal-6.json | recal | 6 | manual-02007 | reject +0.0448 z +1.05 k 213 | - |
| w172-rankreg-2.5-3-i2.json | rankreg | 2.5 | manual-02008 | reject +0.0129 z +1.08 k 111 stopped | - |
| w172-dxp-3.5-lo.json | dxp | 3.5 | manual-02009 | reject +0.0047 z +0.46 k 313 | - |
| w172-rankreg-3-3-i2.json | rankreg | 3 | manual-02010 | reject +0.0256 z +1.23 k 109 stopped | - |
| w172-ens-3-w0.75-R0.25-re.json | ens | 3 | manual-02011 | reject +0.0040 z +0.79 k 313 | - |
| w172-rankreg-3.5-3-i2.json | rankreg | 3.5 | manual-02013 | reject +0.0781 z +1.36 k 88 stopped | - |
| w172-tr-6R0.125.json | tr | 6 | manual-02014 | reject +0.0126 z +0.93 k 313 | - |
| w172-rankreg-4-3-i2.json | rankreg | 4 | manual-02015 | reject -0.0029 z -0.17 k 313 | - |
| w172-ens-4w0.5-R0.5.json | ens | 4 | manual-02016 | reject +0.0077 z +0.34 k 213 | - |
| w172-rankreg-4.5-3-i2.json | rankreg | 4.5 | manual-02017 | tentative -0.0048 z -1.84 k 313 | reject +0.0008 z +0.37 |
| w172-mom-2.json | mom | 2 | manual-02018 | reject +0.0067 z +0.69 k 213 stopped | - |
| w172-rankreg-5-3-i2.json | rankreg | 5 | manual-02019 | reject +0.0154 z +0.94 k 313 | - |
| w172-xplant-2-hi-i3.json | xplant | 2 | manual-02020 | reject -0.0036 z -0.49 k 313 | - |
| w172-ens-2-w0.5-R0.25-i3.json | ens | 2 | manual-02022 | reject +0.0009 z +0.12 k 244 stopped | - |
| w172-anom-attn_outputd1-3.json | anom | 3 | manual-02026 | tentative -0.0154 z -1.51 k 313 | reject +0.0001 z +0.02 |
| w172-xavg-2-w.json | xavg | 2 | manual-02027 | reject -0.0049 z -0.59 k 313 | - |
| w172-dxp-4-both.json | dxp | 4 | manual-02028 | reject +0.0047 z +0.31 k 213 stopped | - |
| w172-xplant-2-hi2.json | xplant | 2 | manual-02030 | reject +0.0345 z +1.32 k 213 | - |
| w172-ens-2-w0.75-R0.25-i3.json | ens | 2 | manual-02032 | reject +0.0629 z +1.65 k 58 stopped | - |
| w172-anom-attn_outputd1-4.json | anom | 4 | manual-02033 | reject -0.0011 z -0.48 k 313 | - |
| w172-xavg-2.5-w.json | xavg | 2.5 | manual-02034 | tentative -0.0077 z -1.61 k 313 | accept -0.0073 z -1.44 |
| w172-rankreg-6-3-i2.json | rankreg | 6 | manual-02035 | reject -0.0018 z -0.11 k 313 | - |
| w172-xplant-2.5-half.json | xplant | 2.5 | manual-02036 | tentative -0.0034 z -1.46 k 313 | reject +0.0133 z +3.45 |
| w172-ens-3.5-w0.5-R0.25-i3.json | ens | 3.5 | manual-02037 | tentative -0.0159 z -2.09 k 313 | reject +0.0113 z +1.72 |
| w172-anom-attn_outputd5-2.5.json | anom | 2.5 | manual-02038 | tentative -0.0129 z -1.61 k 313 | join -0.0101 z -1.84 |
| w172-xavg-2.5.json | xavg | 2.5 | manual-02039 | tentative -0.0153 z -1.51 k 313 | reject -0.0034 z -0.32 |
| w172-ens-3.5-w0.75-R0.25-re.json | ens | 3.5 | manual-02040 | reject +0.0100 z +0.64 k 313 | - |
| w172-xplant-2.5-hi2.json | xplant | 2.5 | manual-02041 | tentative -0.0208 z -2.30 k 313 | reject +0.0021 z +0.21 |
| w172-ens-3.5-w0.5-R0.5-i3.json | ens | 3.5 | manual-02043 | reject +0.0300 z +2.43 k 16 | - |
| w172-anom-attn_outputd5-2.json | anom | 2 | manual-02044 | tentative -0.0094 z -1.57 k 313 | reject -0.0018 z -0.46 |
| w172-xavg-2.json | xavg | 2 | manual-02045 | reject +0.0252 z +1.06 k 213 | - |
| w172-rankreg-2-4-i2.json | rankreg | 2 | manual-02046 | reject +0.0079 z +1.54 k 313 | - |
| w172-xplant-2.5-lo-i3.json | xplant | 2.5 | manual-02047 | reject +0.0263 z +1.37 k 96 | - |
| w172-ens-3.5-w0.75-R0.25-i3.json | ens | 3.5 | manual-02048 | reject +0.0115 z +1.94 k 43 | - |
| w172-anom-attn_outputd5-4.5.json | anom | 4.5 | manual-02049 | tentative -0.0057 z -1.65 k 313 | reject -0.0022 z -0.69 |
| w172-xavg-3-w.json | xavg | 3 | manual-02050 | tentative -0.0168 z -1.56 k 313 | reject +0.0070 z +0.75 |
| w172-tr-6R0.25.json | tr | 6 | manual-02052 | tentative -0.0149 z -0.89 k 313 | reject -0.0005 z -0.04 |
| w172-xplant-2.5-mid-i3.json | xplant | 2.5 | manual-02053 | tentative -0.0129 z -1.89 k 313 | reject -0.0061 z -0.94 |
| w172-ens-4-w0.5-R0.25-i3.json | ens | 4 | manual-02054 | tentative -0.0151 z -0.96 k 313 | reject -0.0030 z -0.20 |
| w172-anom-attn_outputd5-6.json | anom | 6 | manual-02055 | reject +0.0036 z +0.22 k 313 | - |
| w172-xavg-3.5-w.json | xavg | 3.5 | manual-02057 | tentative -0.0122 z -0.84 k 313 | reject +0.0004 z +0.05 |
| w172-rankreg-2.5-4-i2.json | rankreg | 2.5 | manual-02058 | reject +0.0675 z +3.59 k 16 stopped | - |
| w172-xplant-3-half.json | xplant | 3 | manual-02064 | tentative -0.0031 z -0.56 k 313 | reject +0.0009 z +0.18 |
| w172-ens-4-w0.5-R0.5-i3.json | ens | 4 | manual-02066 | reject +0.1002 z +2.25 k 16 stopped | - |
| w172-anom-attn_outputd6-4.5-i3.json | anom | 4.5 | manual-02067 | tentative -0.0045 z -1.40 k 313 | reject -0.0015 z -0.62 |
| w172-xavg-3.5.json | xavg | 3.5 | manual-02069 | reject -0.0074 z -0.77 k 313 | - |
| w172-ens-5w0.5-R0.25.json | ens | 5 | manual-02071 | reject +0.0371 z +1.66 k 109 stopped | - |
| w172-xplant-3-hi-i3.json | xplant | 3 | manual-02072 | tentative -0.0205 z -1.84 k 313 | reject +0.0056 z +0.75 |
| w172-ens-4-w0.75-R0.25-i3.json | ens | 4 | manual-02073 | reject -0.0074 z -0.43 k 313 | - |
| w172-anom-attn_outputd6-6.json | anom | 6 | manual-02074 | reject +0.0047 z +0.31 k 313 | - |
| w172-xavg-3.json | xavg | 3 | manual-02076 | tentative -0.0068 z -0.88 k 313 | reject -0.0021 z -0.30 |
| w172-rankreg-3-4-i2.json | rankreg | 3 | manual-02077 | reject +0.0228 z +1.10 k 109 stopped | - |
| w172-xplant-3-hi2.json | xplant | 3 | manual-02080 | reject +0.0042 z +0.27 k 313 | - |
| w172-ens-4-w0.75-R0.5-i3.json | ens | 4 | manual-02082 | reject +0.0988 z +2.33 k 17 stopped | - |
| w172-anom-ffn_downd0-4.json | anom | 4 | manual-02084 | reject -0.0008 z -0.17 k 313 | - |
| w172-xavg-4-w.json | xavg | 4 | manual-02085 | tentative -0.0090 z -0.70 k 313 | reject -0.0033 z -0.29 |
| w172-floor-3.5.json | floor | 3.5 | manual-02087 | reject +0.1709 z +2.24 k 16 stopped | - |
| w172-xplant-3-lo-i3.json | xplant | 3 | manual-02088 | tentative -0.0143 z -1.16 k 313 | reject +0.0089 z +1.05 |
| w172-ens-4.5-w0.5-R0.25-i3.json | ens | 4.5 | manual-02089 | tentative -0.0097 z -0.91 k 313 | reject -0.0081 z -0.84 |
| w172-anom-ffn_downd6-2.5-i3.json | anom | 2.5 | manual-02090 | reject +0.0098 z +1.76 k 60 | - |
| w172-xavg-4.5-w.json | xavg | 4.5 | manual-02091 | reject +0.0013 z +0.10 k 252 stopped | - |
| w172-rankreg-3.5-4-i2.json | rankreg | 3.5 | manual-02094 | reject +0.1875 z +2.24 k 17 stopped | - |
| w172-xplant-3-lo2.json | xplant | 3 | manual-02095 | reject +0.0483 z +2.27 k 43 stopped | - |
| w172-ens-4.5-w0.5-R0.5-i3.json | ens | 4.5 | manual-02096 | tentative -0.0073 z -0.59 k 313 | reject -0.0015 z -0.15 |
| w172-anom-ffn_downd6-2.json | anom | 2 | manual-02097 | tentative -0.0082 z -1.39 k 313 | reject -0.0005 z -0.13 |
| w172-xavg-4.5.json | xavg | 4.5 | manual-02098 | reject +0.0071 z +0.74 k 313 | - |
| w172-dxp-4-hi.json | dxp | 4 | manual-02099 | reject +0.0277 z +1.54 k 68 stopped | - |
| w172-xplant-3-mid-i3.json | xplant | 3 | manual-02100 | tentative -0.0187 z -1.59 k 313 | reject +0.0036 z +0.45 |
| w172-ens-4.5-w0.75-R0.25-i3.json | ens | 4.5 | manual-02101 | reject +0.0057 z +0.62 k 313 | - |
| w172-anom-ffn_downd6-4.json | anom | 4 | manual-02103 | reject +0.0004 z +0.08 k 313 | - |
| w172-xavg-4.json | xavg | 4 | manual-02105 | reject -0.0072 z -0.44 k 313 | - |
| w172-rankreg-4-4-i2.json | rankreg | 4 | manual-02106 | reject +0.0011 z +0.14 k 240 | - |
| w172-xplant-3.5-half.json | xplant | 3.5 | manual-02107 | reject +0.0225 z +1.29 k 89 stopped | - |
| w172-ens-4.5-w0.75-R0.5-i3.json | ens | 4.5 | manual-02108 | reject +0.0121 z +1.05 k 112 stopped | - |
| w172-anom-ffn_downd7-4.5-i3.json | anom | 4.5 | manual-02109 | tentative -0.0134 z -1.36 k 313 | reject -0.0078 z -0.85 |
| w172-xavg-5-w.json | xavg | 5 | manual-02110 | tentative -0.0105 z -1.00 k 313 | accept -0.0129 z -1.24 |
| w172-ens-4-w0.75-R0.25-re.json | ens | 4 | manual-02111 | reject -0.0076 z -0.40 k 313 | - |
| w172-xplant-3.5-hi-i3.json | xplant | 3.5 | manual-02112 | reject +0.0787 z +1.33 k 84 stopped | - |
| w172-ens-5-w0.5-R0.25-i3.json | ens | 5 | manual-02113 | tentative -0.0275 z -1.77 k 313 | join -0.0127 z -1.01 |
| w172-anom-ffn_downd7-6-i3.json | anom | 6 | manual-02115 | tentative -0.0206 z -1.15 k 313 | reject +0.0175 z +1.45 |
| w172-xavg-5.json | xavg | 5 | manual-02116 | tentative -0.0272 z -1.53 k 313 | reject -0.0007 z -0.08 |
| w172-rankreg-4.5-4-i2.json | rankreg | 4.5 | manual-02117 | tentative -0.0127 z -1.39 k 313 | join -0.0105 z -1.15 |
| w172-xplant-3.5-hi2.json | xplant | 3.5 | manual-02118 | reject +0.0366 z +1.29 k 89 stopped | - |
| w172-ens-5-w0.5-R0.5-i3.json | ens | 5 | manual-02120 | tentative -0.0299 z -1.96 k 313 | reject -0.0080 z -0.95 |
| w172-anom-ffn_gate_upd0-2.5-i3.json | anom | 2.5 | manual-02121 | reject +0.0136 z +1.83 k 44 stopped | - |
| w172-xavg-6-w.json | xavg | 6 | manual-02125 | tentative -0.0128 z -0.72 k 313 | reject +0.0113 z +0.84 |
| w172-xplant-3.5-lo2.json | xplant | 3.5 | manual-02126 | reject +0.0235 z +1.55 k 81 stopped | - |
| w172-ens-5-w0.75-R0.25-i3.json | ens | 5 | manual-02127 | reject +0.0027 z +0.18 k 263 | - |
| w172-anom-ffn_gate_upd0-2-i3.json | anom | 2 | manual-02128 | tentative -0.0131 z -2.11 k 313 | join -0.0044 z -1.12 |
| w172-xavg-6.json | xavg | 6 | manual-02129 | reject +0.0701 z +1.20 k 101 stopped | - |
| w172-rankreg-5-4-i2.json | rankreg | 5 | manual-02130 | tentative -0.0149 z -1.07 k 313 | reject +0.0023 z +0.49 |
| w172-xplant-3.5-mid-i3.json | xplant | 3.5 | manual-02131 | reject +0.0193 z +1.18 k 99 | - |
| w172-ens-6-w0.5-R0.25-i3.json | ens | 6 | manual-02133 | reject +0.0065 z +0.24 k 221 stopped | - |
| w172-anom-ssm_outd4-3-i3.json | anom | 3 | manual-02134 | tentative -0.0042 z -1.08 k 313 | pending |
| w172-ens-5w0.5-R0.5.json | ens | 5 | manual-02136 | reject +0.0364 z +1.19 k 109 | - |
| w172-xplant-4-half.json | xplant | 4 | manual-02137 | reject +0.0004 z +0.03 k 297 stopped | - |
| w172-ens-6-w0.75-R0.25-i3.json | ens | 6 | manual-02141 | reject +0.0128 z +0.69 k 313 | - |
| w172-anom-ssm_outd4-4.json | anom | 4 | manual-02143 | tentative -0.0017 z -0.56 k 313 | pending |
| w172-rankreg-6-4-i2.json | rankreg | 6 | manual-02144 | reject +0.0129 z +1.06 k 313 | - |
| w172-xplant-4-hi-i3.json | xplant | 4 | manual-02146 | reject +0.0203 z +0.61 k 213 stopped | - |
| w172-anom-ssm_outd5-2.5-i3.json | anom | 2.5 | manual-02148 | reject +0.0008 z +0.32 k 213 stopped | - |
| w172-floor-4.json | floor | 4 | manual-02150 | reject +0.0387 z +1.67 k 60 stopped | - |
| w172-xplant-4-hi2.json | xplant | 4 | manual-02152 | reject -0.0010 z -0.07 k 313 | - |
| w172-anom-ssm_outd5-2.json | anom | 2 | manual-02153 | tentative -0.0018 z -1.63 k 313 | pending |
| w172-xplant-4-lo-i3.json | xplant | 4 | manual-02154 | pending | - |
| w172-anom-ssm_zd6-2.5-i3.json | anom | 2.5 | manual-02156 | reject +0.0009 z +0.34 k 213 stopped | - |
| w172-ens-4.5-w0.75-R0.25-re.json | ens | 4.5 | manual-02158 | pending | - |
| w172-xplant-4-lo2.json | xplant | 4 | manual-02160 | pending | - |
| w172-anom-ssm_zd6-2.json | anom | 2 | manual-02161 | pending | - |
| w172-ens-6w0.5-R0.25.json | ens | 6 | manual-02162 | pending | - |
| w172-xplant-4-mid-i3.json | xplant | 4 | manual-02163 | pending | - |
| w172-floor-5.json | floor | 5 | manual-02164 | pending | - |
| w172-xplant-4.5-half.json | xplant | 4.5 | manual-02165 | pending | - |
| w172-xplant-4.5-hi-i3.json | xplant | 4.5 | manual-02167 | pending | - |
| w172-ens-5-w0.75-R0.25-re.json | ens | 5 | manual-02168 | pending | - |
| w172-xplant-4.5-hi2.json | xplant | 4.5 | manual-02169 | pending | - |
| w172-ens-6w0.5-R0.5.json | ens | 6 | manual-02170 | pending | - |
| w172-xplant-4.5-lo-i3.json | xplant | 4.5 | manual-02171 | pending | - |
| w172-floor-6.json | floor | 6 | manual-02172 | pending | - |
| w172-xplant-4.5-lo2.json | xplant | 4.5 | manual-02173 | pending | - |
| w172-xplant-4.5-mid-i3.json | xplant | 4.5 | manual-02174 | pending | - |
| w172-floor-3.json | floor | 3 | manual-02175 | pending | - |
| w172-xplant-5-half.json | xplant | 5 | manual-02176 | pending | - |
| w172-xplant-5-hi-i3.json | xplant | 5 | manual-02177 | pending | - |
| w172-floor-4.5.json | floor | 4.5 | manual-02178 | pending | - |
| w172-xplant-5-lo-i3.json | xplant | 5 | manual-02180 | pending | - |
| w172-xplant-5-lo2.json | xplant | 5 | manual-02182 | pending | - |
