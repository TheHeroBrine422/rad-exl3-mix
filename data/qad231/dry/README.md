# w231 plan231 dry runs (box2, 11-25)
Stand-in sources (the finals' teacher sets tset1 / tset2 and the xgap long docs were not ready): calib qad + fit + search (pools-v3, cap 8k, share 0.5,
minus golden-jlong's ids and qad220's fixed-eval docs), onpol-b1 as "teach" (score_from > 0, share 0.25), xg-scr1 as "xgap" (share 0.25), trdeep-b1 as the
deep traces (12, or 8 via ids). V 24 x Q 1024, 2 ranks, budget 8M rows, lr phase boundary at 3M rows (report only). `summaries.json` = plan231's summary
per spec (doc ids removed), `steps-*.tsv` = per-step cost / carried tokens per rank / deep streams. The plan files themselves stay on box2
(/data/q27/w231/dry: they list trace ids).
