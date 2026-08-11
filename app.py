"""
"명의 vs AI" booth game -- Streamlit app.

Design (see conversation): 5 rounds, visitor taps where they think the abnormality is,
AI's Grad-CAM heatmap is revealed, points for (1) correct diagnosis and (2) tap landing
inside the AI's "hot region". No live model inference here -- everything was
precomputed by prepare_demo_samples.py, so this stays fast even on a phone over wifi.
"""
import json
import os
import random

import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "demo_samples")
UNET_DIR = os.path.join(os.path.dirname(__file__), "unet_examples")
N_ROUNDS = 5

PRIZE_TIERS = [
    (9, "🏆 명의 등급 — 대상"),
    (6, "🩺 레지던트 등급 — 중상"),
    (3, "📖 의대생 등급 — 소정 참가상"),
    (0, "🙂 참가상"),
]


def load_manifest():
    with open(os.path.join(SAMPLES_DIR, "manifest.json"), encoding="utf-8") as f:
        return json.load(f)


def prize_for_score(score, max_score=10):
    for threshold, label in PRIZE_TIERS:
        if score >= threshold:
            return label
    return PRIZE_TIERS[-1][1]


def init_game():
    manifest = load_manifest()
    order = random.sample(manifest, k=min(N_ROUNDS, len(manifest)))
    st.session_state.order = order
    st.session_state.round_idx = 0
    st.session_state.score = 0
    st.session_state.phase = "guess"  # "guess" -> tap to mark -> "reveal" -> show answer
    st.session_state.tap_xy = None
    st.session_state.diag_guess = None


def tap_in_hot_region(tap_xy, hotmask_path):
    if tap_xy is None:
        return False
    mask = Image.open(os.path.join(SAMPLES_DIR, hotmask_path)).convert("L")
    x, y = int(tap_xy["x"]), int(tap_xy["y"])
    x = min(max(x, 0), mask.width - 1)
    y = min(max(y, 0), mask.height - 1)
    return mask.getpixel((x, y)) > 127


def main():
    st.set_page_config(page_title="명의 vs AI", page_icon="🩻")
    st.title("🩻 명의 vs AI — 흉부 X-ray 챌린지")

    if "order" not in st.session_state:
        st.markdown("흉부 X-ray를 보고 **폐렴 여부**를 맞히고, **어디가 이상한지** 짚어보세요. "
                     "AI가 실제로 어딜 보고 판단했는지 바로 비교해드립니다.")

        with st.expander("🔬 이 AI는 어떻게 진단할까? (파이프라인 3단계)", expanded=False):
            st.markdown(
                "**1단계 · 폐 영역 분할 (U-Net)** — X-ray에서 폐 부분만 정확히 찾아냅니다. "
                "(테스트 Dice 0.954, 픽셀 정확도 97.7%)"
            )
            try:
                with open(os.path.join(UNET_DIR, "manifest.json"), encoding="utf-8") as f:
                    unet_examples = json.load(f)
                cols = st.columns(len(unet_examples))
                for col, ex in zip(cols, unet_examples):
                    col.image(os.path.join(UNET_DIR, ex["overlay"]), caption=f"Dice {ex['dice']:.3f}")
            except FileNotFoundError:
                pass

            st.markdown(
                "**2단계 · 폐렴 분류 (전이학습 CNN)** — 사전학습된 신경망으로 정상/폐렴을 판정합니다. "
                "(테스트 정확도 90.9%)\n\n"
                "**3단계 · Grad-CAM 시각화** — 분류기가 실제로 어느 부분을 보고 판단했는지 히트맵으로 보여줍니다. "
                "지금부터 하실 게임이 바로 이 3단계를 직접 확인해보는 겁니다."
            )

        if st.button("시작하기", type="primary"):
            init_game()
            st.rerun()
        return

    if st.session_state.round_idx >= len(st.session_state.order):
        score = st.session_state.score
        max_score = len(st.session_state.order) * 2
        st.header(f"최종 점수: {score} / {max_score}")
        st.subheader(prize_for_score(score, max_score))
        if st.button("다시 하기"):
            init_game()
            st.rerun()
        return

    sample = st.session_state.order[st.session_state.round_idx]
    st.caption(f"라운드 {st.session_state.round_idx + 1} / {len(st.session_state.order)}"
               f"   |   현재 점수: {st.session_state.score}")

    if st.session_state.phase == "guess":
        st.subheader("이 흉부 X-ray, 정상일까요 폐렴일까요?")
        st.caption("사진에서 이상해 보이는 부분을 탭하고, 아래에서 진단도 골라주세요. 둘 다 고르면 다음으로 넘어갑니다.")

        img = Image.open(os.path.join(SAMPLES_DIR, sample["plain_image"]))
        coords = streamlit_image_coordinates(img, key=f"tap_{st.session_state.round_idx}")
        if coords is not None:
            st.session_state.tap_xy = coords

        if st.session_state.tap_xy is not None:
            st.caption("탭 위치 저장됨 ✅")

        col1, col2 = st.columns(2)
        with col1:
            normal_type = "primary" if st.session_state.diag_guess == "normal" else "secondary"
            if st.button("정상", use_container_width=True, type=normal_type):
                st.session_state.diag_guess = "normal"
                st.rerun()
        with col2:
            pneumonia_type = "primary" if st.session_state.diag_guess == "pneumonia" else "secondary"
            if st.button("폐렴", use_container_width=True, type=pneumonia_type):
                st.session_state.diag_guess = "pneumonia"
                st.rerun()

        if st.session_state.diag_guess is not None and st.session_state.tap_xy is not None:
            st.session_state.phase = "reveal"
            st.rerun()

    elif st.session_state.phase == "reveal":
        diag_correct = st.session_state.diag_guess == sample["true_label"]
        tap_correct = tap_in_hot_region(st.session_state.tap_xy, sample["hot_mask"])
        round_score = int(diag_correct) + int(tap_correct)
        st.session_state.score += round_score

        st.image(os.path.join(SAMPLES_DIR, sample["overlay_image"]),
                  caption="빨간색이 진할수록 AI가 집중해서 본 부분입니다")

        st.write(f"**정답**: {sample['true_label']} (당신의 답: {st.session_state.diag_guess}) "
                 f"{'✅' if diag_correct else '❌'}")
        st.write(f"**AI 예측 확률**: {sample['predicted_prob']:.1%} 폐렴")
        st.write(f"**위치 매칭**: {'✅ AI랑 같은 곳을 봤어요!' if tap_correct else '❌ AI는 다른 곳을 봤어요'}")
        st.write(f"이번 라운드 점수: **+{round_score}**")

        if st.button("다음 라운드", type="primary"):
            st.session_state.round_idx += 1
            st.session_state.phase = "guess"
            st.session_state.tap_xy = None
            st.session_state.diag_guess = None
            st.rerun()


if __name__ == "__main__":
    main()
