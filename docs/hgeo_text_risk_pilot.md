# H_geo: SUN 이름 기반 위험 예측 — 작은 탐색 실험

작성일: 2026-09-28. 브랜치: `hgeo-text-risk-pilot`.
기준 브랜치: `stage4-b16-h1-replication`.

## 질문과 범위

H_geo는 hypothesis + geometry의 작업용 이름이며 확립된 방법명이 아니다.
여기서는 **장면 이름만으로 계산한 위험 순위가, 그 장면 이미지의 실제
OOD 오탐률 순위와 맞는가?**만 시험한다.
예를 들어 river와 ocean 중 어떤 개념에서 OOD 탐지가 더 많이 실패할지를
이름으로 계산하고 기존 이미지 결과와 비교한다. 예측 방향은 결과를 보기
전에 코드로 고정하며, 특정 개념의 실제 성공/실패를 미리 가정하지 않는다.

이것은 첨부된 큰 수정안 전체가 아니라 최소 범위의 별도 탐색이다.
기존 H1/H2 가설, FAIL 판정, subgroup, eligibility, 임계값을 변경하지 않는다.
기존 결과를 이미 검토했으므로 새 브랜치/새 코드여도 확증 연구가 되지 않는다.

## 고정한 첫 시험

| 항목 | 설정 |
| --- | --- |
| 백본 | OpenAI CLIP ViT-B/32 |
| OOD | SUN, 커밋된 50개 leaf 이름을 수정하지 않음 |
| 탐지기 | 기존 MCM, NegLabel 구현 |
| 이름 prompt | `a photo of a {}` 한 개, 두 방법에 동일 적용 |
| 실제 오류 | 기존 reproduction CSV의 score >= 기존 ID95 기준 |
| 임계값 | 기존 `build_fixed_id_reference`: ID CSV의 5% higher 분위수, 방법별 한 개 |
| 분석 단위 | 개념, n>=20 이미지; 조건당 적격 개념 10개 이상 필요 |
| 주 탐색 지수 | 개념명 텍스트 임베딩을 이미지 임베딩 대신 기존 탐지기 점수식에 입력 |
| 비교 지수 | 두 방법 각각의 ID 텍스트와 최대 cosine similarity |
| 보조 지수 | MCM: max-minus-mean peak; NegLabel: -max negative coverage, ID q95 |
| 요약 | Spearman rho, 개념 2000회 bootstrap 95% CI, seed=42 |
| 기준선 비교 | 같은 bootstrap 표본에서 rho(text proxy)-rho(max ID cosine) |

NegLabel의 기존 negative 목록, 순서, 100개 grouping, logit scale, temperature를
그대로 사용한다. 새 mining, negative 복원, prompt 최적화는 하지 않는다.
보조 coverage는 낮을수록 위험하다는 방향을 반영해 음수를 사용한다.
coverage와 q95가 실패해도 주 지수를 사후 교체하지 않는다.

이미지/이미지 feature를 새로 읽거나 인코딩하지 않으며 학습도 하지 않는다.
CLIP checkpoint와 텍스트 encoding, 저장된 baseline CSV 및 의미 매핑만 필요하다.
`--backbone ViT-B/16`은 같은 탐색의 선택적 반복이며 독립 확증으로 세지 않는다.
학명 공통명 변환, iNaturalist, NINCO, 신규 백본, 인과 개입, H2/H3는 제외한다.

## 실행

새 clone이 아니라 기존 데이터와 미커밋 raw score가 있는 checkout을 사용한다.
로컬 변경을 자동 stash/reset하지 않는다. switch가 거부되면 먼저 변경 사항을 확인한다.

```bash
cd ~/clip-ood-hidden-failure
conda activate clip-ood
git status --short
git fetch origin
git switch hgeo-text-risk-pilot

pytest -q tests/test_hgeo_text_risk_pilot.py
python scripts/analysis/run_hgeo_text_risk_pilot.py --check-inputs
python scripts/analysis/run_hgeo_text_risk_pilot.py --gpu 0
```

필요한 로컬 파일:

```text
results/raw/reproduction/ViT-B-32/{mcm,neglabel}/imagenet.csv
results/raw/reproduction/ViT-B-32/{mcm,neglabel}/sun.csv
${CLIP_OOD_DATA_ROOT:-data}/semantic_labels/sun_image_semantic_labels.csv
configs/subgroups/mappings/sun_mos50_hierarchy.csv
```

기존 환경의 CLIP weights와 `third_party/{MCM,NegLabel}`도 필요하다.
B/32의 기존 mined negatives가 없으면 기존 loader가 오류를 내며 자동 재생성하지 않는다.
`--check-inputs`는 CSV만 검사한다. GPU/weights/reference repos 검증 성공을 의미하지 않는다.

출력은 기존 결과와 분리한다. 기존 출력 폴더가 있으면 덮어쓰지 않고 중단한다.
의도한 재실행은 새 `--run-name sun_pilot_v2`로 원인을 함께 기록한다.

```text
results/hgeo_text_risk/ViT-B-32/sun_pilot_v1/
  concept_results.csv
  correlation_summary.csv
  pilot_summary.json
```

## 읽는 법과 중단 범위

먼저 각 방법의 `text_proxy_score` 행을 읽는다. rho>0은 이름 위험 순위가 실제
오탐 순위와 같은 방향이라는 뜻이다. rho>=0.3 및 CI 하한>0이면
`PROMISING_DESCRIPTIVE_SIGNAL`로 표시한다. 이 0.3 기준은 탐색 우선순위용
휴리스틱이며 연구 PASS, 유의성 확증, 독창성 판정이 아니다.
`NO_CLEAR_POSITIVE_SIGNAL`은 이 조건에서 뚜렷한 양의 신호가 없다는 뜻이고,
상수 입력/불안정 bootstrap은 `UNDEFINED_OR_UNSTABLE`로 구분한다.

`delta_rho_vs_max_id` 및 paired CI를 함께 확인한다. 양의 상관만 있고 단순
기준선보다 낫지 않으면 detector-specific 추가 가치가 확인된 것이 아니다.
차이 CI가 0을 포함하면 무조건 같다고 판정하지 않는다.

CI는 관측된 이미지와 ID reference에 조건부인 개념 단위 기술 통계다.
이미지/ID 샘플링 불확실성, 개념 사이 계층 의존성, 다중비교를 모두 보정하는
확증 CI가 아니다. 이름 점수에 ID threshold를 적용하거나 이를 실제 FPR 확률로 읽지 않는다.
기존 결과를 이미 검토한 상태라는 탐색 한계도 남는다.

두 방법 모두 신호가 약하면 이 SUN/이름 표현의 최소 시험을 종료 대상으로
기록한다. 이름 선택이나 보조 지수를 바꾸어 같은 시험을 성공으로 다시
표시하지 않는다. 이 한 소스의 부정적 결과를 모든 H_geo의 기각으로 확대하지 않는다.
유망하더라도 별도 데이터 확증/선행연구 비교 없이 논문가치를 확정하지 않는다.
mining이 원인인지, negative 복원이 효과적인지는 이번 시험으로 알 수 없다.

## 구현 검증 상태

제공 환경에서 synthetic/CPU 단위 테스트 16개 통과.
실제 CLIP checkpoint, CUDA, 사용자의 로컬 CSV를 이용한 실행은 아직 하지 않았다.
테스트 데이터는 연구 결과가 아니며 output에 실제 실험 수치를 미리 넣지 않았다.

## 다음 채팅 handoff

채팅 이름: `CLIP-OOD 08 — H_geo 최소 탐색 실행 결과`.

완료: hgeo-text-risk-pilot 브랜치, 독립 runner/프로토콜/CPU 테스트 추가; 16 tests passed.
확정: SUN+ViT-B/32, MCM/NegLabel의 실제 점수식을 텍스트에 적용, 원래 ID95와 H1/H2 판정 유지.
핵심 수치: synthetic tests 16 통과; bootstrap 2000; 실제 rho/FPR 결과는 아직 없음.
미완료: 사용자 CUDA/로컬 CSV 실행, 새로운 개념 확증, mining 인과 시험, 논문가치 확정.
다음 첫 작업: pilot_summary.json과 correlation_summary.csv로 입력/임계값/완료 상태를
먼저 확인하고 주 지수와 max-ID 기준선을 비교한다. 결과를 보고 지수나 부호를 바꾸지 않는다.
