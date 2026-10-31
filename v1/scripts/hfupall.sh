#!/bin/bash
# the nine existing (SC) repos, overwritten one at a time
cd /root
/root/hfup.sh sc1.40-e3 qwen3.8-27b-exl3-1.40bpw SC_1.40bpw_H3 3
/root/hfup.sh sc1.60-e3 qwen3.8-27b-exl3-1.60bpw SC_1.60bpw_H3 3
/root/hfup.sh sc1.80-e3 qwen3.8-27b-exl3-1.80bpw SC_1.80bpw_H3 3
/root/hfup.sh sc2.00-e3 qwen3.8-27b-exl3-2.00bpw SC_2.00bpw_H3 3
/root/hfup.sh sc2.20-e3 qwen3.8-27b-exl3-2.20bpw SC_2.20bpw_H3 3
/root/hfup.sh sc3.00-e4 qwen3.8-27b-exl3-3.00bpw SC_3.00bpw_H4 4
/root/hfup.sh sc4.00-e4 qwen3.8-27b-exl3-4.00bpw SC_4.00bpw_H5 4
/root/hfup.sh sc5.00-e5 qwen3.8-27b-exl3-5.00bpw SC_5.00bpw_H6 5
/root/hfup.sh sc6.00-e6 qwen3.8-27b-exl3-6.00bpw SC_6.00bpw_H6 6
echo HFDONE >> /root/hfup.log
