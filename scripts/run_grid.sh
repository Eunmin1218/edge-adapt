#!/usr/bin/env bash
# 본 실험 전체: D(선별 4종 x 예산 3단계 x 시드 3) + A, T, C(tent/bn) + B, AMS + Ablation + 레지스트리 OFF.   tmux 안에서 실행:  bash scripts/run_grid.sh
# 환경변수로 덮어쓴다:  BUDGETS="16" SEEDS="0" DELAY=300 bash scripts/run_grid.sh   (노트북 CPU 서버처럼 느린 환경에서 줄여 돌릴 때)
set -e
BUDGETS=${BUDGETS:-"8 16 32"}; DELAY=${DELAY:-100}; SEEDS=${SEEDS:-"0 1 2"}; R="python scripts/run_experiment.py --split eval"
$R --method A --tag A; $R --method T --tag T; $R --method C --tent tent --tag C_tent; $R --method C --tent bn --tag C_bn
for s in $SEEDS; do
  for b in $BUDGETS; do
    $R --method B   --budget $b --delay $DELAY --seed $s --tag B_b${b}_s${s}
    $R --method AMS  --budget $b --delay $DELAY --seed $s --tag AMS_b${b}_s${s}
    $R --method RECL --budget $b --delay $DELAY --seed $s --tag RECL_b${b}_s${s}
    $R --method ES   --budget $b --delay $DELAY --seed $s --tag ES_b${b}_s${s}
    for sel in random uncertainty diversity hybrid entropy ent_hybrid; do
      $R --method D --selector $sel --budget $b --delay $DELAY --seed $s --tag D_${sel}_b${b}_s${s}
    done
  done
  $R --method D --selector diversity --budget 16 --cols conf --delay $DELAY --seed $s --tag Dconf_diversity_b16_s${s}
  $R --method D --selector diversity --budget 16 --cols temp --delay $DELAY --seed $s --tag Dtemp_diversity_b16_s${s}
  for th in 0.08 0.25; do $R --method ES --budget 16 --theta $th --delay $DELAY --seed $s --tag ES_th${th}_b16_s${s}; done      # 비교 대상의 임계값을 튜닝해도 지는지 확인
  $R --method D --selector diversity --budget 16 --consec 2 --delay $DELAY --seed $s --tag Dconsec2_diversity_b16_s${s}                  # 감지 확정을 2개 윈도우 연속 확인으로 늘림(오경보 억제 vs 반응 속도)
  $R --method D --selector edgesync --budget 16 --delay $DELAY --seed $s --tag Dsel_edgesync_b16_s${s}                         # 제안 파이프라인 + EdgeSync 선별 (선별 기여 분리)
  $R --method D --selector diversity --budget 16 --no-registry --delay $DELAY --seed $s --tag Dnoreg_diversity_b16_s${s}
  for tau in 0.3 0.7; do $R --method D --selector diversity --budget 16 --tau $tau --delay $DELAY --seed $s --tag Dtau${tau}_diversity_b16_s${s}; done
done
python scripts/analyze.py
