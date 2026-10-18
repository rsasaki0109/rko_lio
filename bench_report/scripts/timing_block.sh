#!/usr/bin/env bash
# All timing runs, one at a time. Frontend-only exp04 ABAB, then the full pipeline.
A=/home/user/bench/abab
W=/home/user/ws
echo "start $(date -u +%H:%M:%S)"
/home/user/scripts/abab.sh $A/adj1_80f3c1c_12cc517.csv 3 80f3c1c $W/base_80f3c1c 12cc517 $W/stack_12cc517
/home/user/scripts/abab.sh $A/adj2_12cc517_0250b4d.csv 3 12cc517 $W/stack_12cc517 0250b4d $W/stack_0250b4d
/home/user/scripts/abab.sh $A/adj3_0250b4d_4dd6094.csv 3 0250b4d $W/stack_0250b4d 4dd6094 $W/stack_4dd6094
/home/user/scripts/abab.sh $A/adj4_4dd6094_0681b06.csv 3 4dd6094 $W/stack_4dd6094 0681b06 $W/stack_0681b06
/home/user/scripts/abab.sh $A/adj5_0681b06_f6b0425.csv 3 0681b06 $W/stack_0681b06 f6b0425 $W/stack_f6b0425
echo "adjacent done $(date -u +%H:%M:%S)"
/home/user/scripts/abab.sh $A/new1_f6b0425_f4a6547.csv 3 f6b0425 $W/stack_f6b0425 f4a6547 $W/cand_f4a6547
/home/user/scripts/abab.sh $A/new2_f4a6547_734d54f.csv 3 f4a6547 $W/cand_f4a6547 734d54f $W/cand_734d54f
/home/user/scripts/abab.sh $A/new3_f6b0425_734d54f.csv 5 f6b0425 $W/stack_f6b0425 734d54f $W/cand_734d54f
echo "frontend done $(date -u +%H:%M:%S)"
P=/home/user/bench/pipe
/home/user/scripts/pipe_abab.sh $P/pipe1_base_f6b0425.csv 3 80f3c1c $W/pipe_base f6b0425 $W/pipe_f6b0425
/home/user/scripts/pipe_abab.sh $P/pipe2_f6b0425_734d54f.csv 3 f6b0425 $W/pipe_f6b0425 734d54f $W/pipe_734d54f
echo "pipeline done $(date -u +%H:%M:%S)"
# Diagnostic (not part of the RTF table): per-process CPU of the frontend and the graph backend.
for lab in f6b0425 734d54f; do
  sleep 20
  /home/user/scripts/sample_proc_cpu.sh /home/user/bench/pipe/diag_cpu_${lab}.csv 90 &
  sampler=$!
  /home/user/scripts/pipe_once.sh /home/user/bench/pipe/diag_${lab}.csv diag_${lab} $W/pipe_${lab} > /dev/null 2>&1 &
  runner=$!
  wait ${sampler}
  wait ${runner}
done
echo "diag done $(date -u +%H:%M:%S)"
