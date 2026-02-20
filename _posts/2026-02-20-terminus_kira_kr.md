---
layout: post
title: "[KR] Terminus-KIRA로 terminal-bench에서 74.8%를 달성한 방법"
description: Terminus-KIRA를 소개합니다. 최소한의 하네스 개선으로 terminal-bench에서 프론티어 모델 성능을 향상시켰습니다.
date: 2026-02-20
hidden: false
---

<img src="{{ 'assets/img/2026-02-20-terminus_kira/terminus_kira_1.jpeg' | relative_url }}" style="width: 100%; height: auto;">

## 1. 사전 지식: terminal-bench란? Terminus란?

***terminal-bench***는 AI 에이전트가 터미널 환경에서 실제 작업을 얼마나 잘 수행하는지 평가하는 벤치마크입니다. "터미널"이라고 하지만, 흥미로운 수학/ML 문제들이 많이 포함되어 있어서 기본적으로 "AI 에이전트가 ML/AI/SW 엔지니어가 하는 일을 할 수 있는가 :P?"라는 질문입니다. 터미널 기반 작업은 디버깅, 코딩, 실험 실행, 그리고 수정 사항의 end-to-end 배포를 포함한 실제 엔지니어링의 상당 부분이 이루어지는 곳이기 때문에 중요합니다. 


***Terminus***는 LLM이 실제 터미널(tmux를 통해)과 상호작용하고, 출력을 관찰하며, 최종 답변을 제출하기로 결정할 때까지 반복할 수 있게 해주는 에이전트 하네스 입니다. 다양한 도구를 통해 가상 머신과 직접 상호작용하는 다른 "in-machine" 하네스들과 비교했을 때, Terminus는 의도적으로 미니멀합니다: "명령어 전송 → 버퍼 읽기 → 생각 → 반복"이라는 깔끔한 루프입니다.

대부분의 frontier AI 연구소들은 자사 모델에 Terminus 2를 탑재하고 terminal-bench에서의 평가 결과를 보고합니다. 개념적으로 매우 단순하고 우아하며, 충분히 좋은 에이전트라면 결국 이것만으로 최대 정확도를 달성할 수 있을 것이라고 믿고 싶어집니다.

하지만 좀 고쳐야 할 것들이 있습니다…

## 2. 현실

하지만 실제로 다양한 에이전트를 terminal-bench에서 실행하고 어떻게 실패하는지 관찰하면, 매우 흥미로운 실패 양상들을 발견할 수 있습니다. 저희는 자체 에이전트 행동 분석기를 개발했습니다. 즉, 에이전트가 무엇을 잘하고, 무엇을 못하는지, 그리고 성공/실패 행동을 분석하는 자동화된 분석 도구입니다.


분석 결과는 꽤 충격적입니다.

**TL;DR:** Terminus 2가 극도로 미니멀한 하네스이기 때문에, 매우 뛰어난 모델들도 피할 수 있었을 실수들을 수없이 저지릅니다.

다음은 Terminus 2 + 프론티어 모델 분석에서 얻은 가장 중요한 시사점들입니다.

### 2-1. 모델은 인간을 "보조"하도록 최적화되어 있지, 인간을 완전히 대체하도록 최적화되어 있지 않습니다

이것이 이 보고서에서 가장 전달하고 싶은 메시지라고 생각합니다. 모델은 대부분 우리(우리 = 인간)와 상호작용하며 우리를 "보조"하도록 훈련되었지, 전체 작업을 스스로 완수하도록 훈련되지 않았습니다.

**- 모델은 부분적인 작업을 제출하고 "일단 해보는" 경향이 있습니다**

이것은 전통적인 "assistant" 일때는 전혀 문제가 없으며, 실제로 바람직한 행동이었습니다. 모델이 부분적인 결과를 보여주면 우리(인간)가 초기에 피드백을 줄 수 있기 때문입니다. 하지만 에이전트가 전체 작업을 완벽하게 완수하기를 기대하는 terminal-bench(또는 longer-horizon task)에서는 좋지 않습니다. 이러한 모델을 완수 방향으로 약하게만 유도하는 에이전트 루프에 넣어도, 특히 매우 어렵고 긴 호라이즌 계획이 필요한 작업에서는 여전히 반쯤 완성된 결과를 반환하는 경향이 있습니다. Terminus 2는 너무 미니멀해서 모델이 주어진 작업을 높은 확률로 완수하도록 유도하는 데 실패합니다.

**- 인간에게 시각적 결과물을 보여주려는 경향이 있습니다**

이것은 제가 가장 좋아하는 발견입니다. 최신 LLM들은 모두 "눈"을 가지고 있으며, 자신이 눈을 가지고 있다고 믿도록 post-training되어 있습니다. 하지만 여전히 우리를 보조하도록 훈련되었기 때문에, 복잡한 시각적 검사나 이해가 필요한 작업에서 모델은 그 어려운 시각적 이해 부분을 우리에게 넘기는 경향이 있습니다, 으악!!! Terminus 2는 이러한 떠넘기기 행동을 방지하지 않습니다.

### 2-2. 모델은 아직 완벽하지 않습니다

**- 나쁜 자기 평가 (거짓 완료)**

모델은 자신의 출력을 자기 평가하는 데 정말 못합니다. 모델이 끝났다고 생각하면, Terminus 2는 "정말 확실해?"라고 다시 물어보도록 설계되어 있습니다. 그러면 보통 작업이 완료되지 않았거나 틀렸더라도 "응, 확실해 :-)"라고 답합니다. 이로 인해 많은 "거짓 완료" 오류가 발생합니다.

**- 나쁜 적응적 재계획**

오늘날의 프론티어 모델에는 진정한 병목현상이 있습니다. 모든 정보가 주어진 상태에서 처음부터 계획을 세우는 것은 꽤 잘하지만, 새로운 정보를 관찰한 후 기존 계획을 조정하는 데는 어려움을 겪습니다.

### 2-3. Terminal-bench 및 Terminus 고유의 실패 양상

예를 들어, 솔루션이 얼마나 일반적이어야 하는지에 대해 문제가 모호한 경우가 있습니다. 에이전트가 임의로 다른 환경에서도 작동하는 완벽한 솔루션을 만들어야 하는지, 아니면 현재 환경에서 잘 실행되는지만 확인하면 되는지가 명확하지 않습니다. 에이전트는 더 좁은 의미로 질문에 답하는 솔루션을 찾는 경향이 있습니다. 이것은 시험 중에 "좋은" 학생이라면 손을 들고 명확히 할 기회를 가졌을 질문입니다! 하지만 에이전트에게는 그런 여유가 없습니다. 이로 인해 에이전트는 좁고 엄격한 문제 사양 사이에서 임의로 추측하게 됩니다.



TerminalBench2 규칙상, 에이전트는 자신에게 얼마나 많은 시간이 주어지는지 모릅니다. 이 때문에 일부 에이전트는 거대한 라이브러리와 도구를 설치하여, 의존성 설치에 너무 많은 시간을 낭비합니다.

**Terminus에는 두 가지 중요한 고유 한계점도 있습니다:**

- Terminus는 "push and wait" 메커니즘을 사용합니다. 즉, 기반 tmux에 명령을 보내고 "추정된" 실행 시간만큼 기다립니다. 이로 인해 무시할 수 없는 시간 낭비가 발생합니다.

- Terminus는 내부 tmux 화면을 사용하여 버퍼 출력을 읽어 LLM에 전달합니다. 기본 버퍼 크기가 너무 작아서, 큰 파일을 읽을 때 에이전트가 혼란을 겪습니다.

## 3. Terminus-KIRA 소개

위에서 언급한 문제들에 대해 몇 가지 간단한 수정을 제안합니다.



첫 번째 문제를 해결하기 위해, 에이전트 프롬프트를 몇 가지 변경하고 도구를 하나 더 추가했습니다.

- 에이전트 프롬프트에 인간과의 상호작용 없이 단 한 번의 제출만 가능하며, 제출은 FINAL이라는 것을 명확히 명시했습니다. 이를 통해 에이전트가 성급한 답변을 제출하는 것을 방지합니다.

- 에이전트가 인간에게 시각적 검사를 기대하는 것을 방지하기 위해, 문자 그대로 다음 프롬프트를 추가했습니다: "You must complete the entire task without any human intervention."

- 그리고 tmux 버퍼가 멀티미디어 파일을 전달할 수 없기 때문에, 멀티미디어 파일 읽기/이해를 위해 특별히 지정된 도구를 추가했습니다. 따라서 LLM에게 두 가지 도구가 주어집니다: (1) 명령 실행, 또는 (2) 백엔드 LLM을 사용하여 멀티미디어 파일을 직접 읽기/이해하기. 또한 다음 프롬프트도 추가했습니다: "You do NOT have eyes or ears, so you MUST resort to various programmatic/AI tools to understand multimedia files."
<img src="{{ 'assets/img/2026-02-20-terminus_kira/terminus_kira_2.jpeg' | relative_url }}" style="width: 100%; height: auto;">

두 번째 문제(나쁜 자기 평가와 나쁜 재계획)도 해결하고자 했습니다.

- 자기 완료 검증 부분에서 이제 매우 철저한 단계별 객관적 자기 평가를 요구합니다. 거짓 완료 비율이 상당히 감소한 것을 확인했습니다.
<img src="{{ 'assets/img/2026-02-20-terminus_kira/terminus_kira_3.jpeg' | relative_url }}" style="width: 100%; height: auto;">
- 재계획에 대해서는 매우 간단한 프롬프팅 기법을 사용하여 모델이 적응적으로 더 잘 재계획하도록 도왔습니다.



마지막으로, 프롬프트에 몇 가지 일반적인 팁도 추가했습니다. 예를 들면:
<img src="{{ 'assets/img/2026-02-20-terminus_kira/terminus_kira_4.jpeg' | relative_url }}" style="width: 100%; height: auto;">
첫 번째 팁은 에이전트가 더 넓고 일반적인 솔루션을 만들도록 돕습니다. 두 번째 팁은 에이전트가 무거운 의존성을 설치하여 타임아웃 오류를 일으키는 것을 방지합니다.

마지막으로, tmux 인터페이스도 수정했습니다. tmux 인터페이스가 "pull" 메커니즘으로 업데이트되었습니다. 이제 예측된 실행 시간이 실제 실행 시간보다 큰 경우 에이전트가 초과 시간을 기다릴 필요가 없습니다. tmux 버퍼 크기도 늘렸습니다.

## 4. 결과?

잘 됩니다 :)

<img src="{{ 'assets/img/2026-02-20-terminus_kira/terminus_kira_5.jpeg' | relative_url }}" style="width: 100%; height: auto;">

저희는 Terminus-KIRA를 오픈소스로 공개하기로 했습니다.
https://github.com/krafton-ai/kira

즐겨주세요 :)

## 5. Takeaway & prediction

**TL;DR:**

- Terminus-KIRA는 프론티어 모델의 성능을 10% 포인트 향상시킵니다
- 저희의 수정은 매우 간단하지만 매우 효과적입니다.
- Terminus-KIRA를 사용해보세요.

그리고 현재 모델들에 더 나은 하네스만 적용해도 TerminalBench2에서 조만간 80%를 넘길 것이라고 확신합니다. 제 100원 걸겠습니다 :-)

---

## Acknowledgement

*All authors are listed alphabetically by last name.*

**Agent Design & Experiments**<br>
Minseok Choi<br>
Wooseong Chung<br>
Yun Jegal<br>
Jiho Jeon<br>
Giyoung Jung<br>
Seungjin Kwon<br>
Gisang Lee<br>
Hyogon Ryu

**Project Coordination**<br>
Myungseok Oh

**Infrastructure**<br>
Hara Kang

**Advising & Writing**<br>
Kangwook Lee

---

## Citing Us

If you found Terminus-KIRA useful, please cite us as:

```bibtex
@misc{terminuskira2026,
      title={Terminus-KIRA: Terminus-KIRA: Boosting Frontier Model Performance on Terminal-Bench with Minimal Harness },
      author={Minseok Choi and Wooseong Chung and Yun Jegal and Jiho Jeon and Giyoung Jung and Seungjin Kwon and Gisang Lee and Hyogon Ryu and Myungseok Oh and Hara Kang and Kangwook Lee},
      year={2026},
      url={https://github.com/krafton-ai/kira},
}
```

---
**KRAFTON AI & Ludo Robotics**