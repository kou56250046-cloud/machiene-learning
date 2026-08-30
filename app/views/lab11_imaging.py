"""ラボ 11 — 画像と信号。

画像は数値の行列、信号は数値の並び。どちらも「畳み込み」と「周波数」という
共通の道具で扱える。CNN の中身がここにある。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app.components import experiment_log as EL
from app.components import imaging_plots as P
from app.components import tabular_plots as TabP
from app.components.cards import Kpi, kpi_row, score_color
from app.components.explain import explain
from app.components.layout import note, page_header, panel, sidebar_section
from mllab.models import imaging as IM
from mllab.models import signals as SG
from mllab.viz import theme

KEY = "lab11"

accent = page_header(
    number=11,
    title="画像・信号ラボ",
    lede=(
        "画像は数値が縦横に並んだ行列、信号は数値が時間順に並んだもの。"
        "小さな行列を滑らせて掛け合わせる「畳み込み」が CNN の中身であり、"
        "波を周波数に分解する「フーリエ変換」が音声解析の土台です。"
        "どちらも自分で組んで、何が起きるかを直接確かめられます。"
    ),
)

# 分類タブで求めた成績を、ページ末尾の記録欄でも使うため外に置いておく
digit_scores: dict[str, float] = {}

tabs = st.tabs(["画像は数値の行列", "畳み込みフィルタ", "特徴量と分類", "信号と周波数"])

# ======================================================================
# タブ 1: 画像は数値の行列
# ======================================================================
with tabs[0]:
    sidebar_section("画像")
    image_key = st.sidebar.selectbox(
        "サンプル画像", list(IM.IMAGES),
        format_func=lambda k: IM.IMAGES[k].label, key=f"{KEY}_image",
    )
    st.sidebar.caption(IM.IMAGES[image_key].note)

    has_color = IM.is_color(image_key)
    use_gray = st.sidebar.toggle(
        "グレースケールにする", value=True, key=f"{KEY}_gray",
        disabled=not has_color,
        help="カラー画像だけ切り替えられます。畳み込みは 3 チャンネルにも当たります。",
    )

    @st.cache_data(show_spinner=False)
    def load(key: str, grayscale: bool) -> np.ndarray:
        return IM.load_image(key, grayscale=grayscale)

    image = load(image_key, use_gray or not has_color)

    kpi_row(
        [
            Kpi("大きさ", f"{image.shape[1]} × {image.shape[0]}", sub="幅 × 高さ（画素）"),
            Kpi("チャンネル", "3（RGB）" if image.ndim == 3 else "1（濃淡）",
                sub="カラーは 3 枚重なっています"),
            Kpi("画素数", f"{int(np.prod(image.shape[:2])):,}",
                sub="そのまま特徴量にすると次元数になります", color=theme.PURPLE),
            Kpi("画素値の範囲", f"{image.min():.2f} 〜 {image.max():.2f}",
                sub="0 が黒、1 が白", color=theme.ORANGE),
        ],
        accent,
    )

    col_image, col_hist = st.columns([1.3, 1])
    with col_image:
        panel("画像そのもの", IM.IMAGES[image_key].label)
        st.plotly_chart(
            P.image_figure(image, height=380), width="stretch", key=f"{KEY}_img"
        )
    with col_hist:
        panel("画素値の分布", "明るさがどこに偏っているか")
        st.plotly_chart(
            P.histogram_figure(image, height=250), width="stretch", key=f"{KEY}_hist"
        )
        st.caption(
            "山が右に寄っていれば明るい画像、左なら暗い画像です。"
            "2 つの山に分かれていれば、明るい部分と暗い部分がはっきり分かれています。"
        )

    if image.ndim == 3:
        panel("カラーは 3 枚の重ね合わせ", "赤・緑・青それぞれの明るさ")
        st.plotly_chart(
            P.channel_figure(image), width="stretch", key=f"{KEY}_channels"
        )
        st.caption(
            "同じ位置の 3 つの値を混ぜると元の色になります。"
            "モデルにとってカラー画像は「3 枚重なった行列」でしかありません。"
        )

    panel("拡大して数値を見る", "画像が本当にただの数値であることを確かめます")

    gray_for_patch = image if image.ndim == 2 else image.mean(axis=2)
    height, width = gray_for_patch.shape
    col_x, col_y, col_size = st.columns(3)
    patch_size = col_size.slider("切り出す大きさ", 6, 16, 10, 1, key=f"{KEY}_patch")
    x0 = col_x.slider("横の位置", 0, max(0, width - patch_size), width // 3, 1,
                      key=f"{KEY}_px")
    y0 = col_y.slider("縦の位置", 0, max(0, height - patch_size), height // 3, 1,
                      key=f"{KEY}_py")
    patch = gray_for_patch[y0 : y0 + patch_size, x0 : x0 + patch_size]

    st.plotly_chart(P.pixel_grid_figure(patch), width="stretch", key=f"{KEY}_grid")
    st.caption(
        f"{patch_size}×{patch_size} = {patch_size**2} 個の数値です。"
        "輪郭のあるところを切り出すと、隣り合う数値が大きく変わっているのが分かります。"
        "**畳み込みはこの「隣との差」を拾う操作です。**"
    )

# ======================================================================
# タブ 2: 畳み込みフィルタ
# ======================================================================
with tabs[1]:
    panel("カーネルを当ててみる", "3×3 の小さな行列を、画像全体に滑らせながら掛け合わせます")

    col_choice, col_kernel = st.columns([1, 1])
    with col_choice:
        kernel_key = st.selectbox(
            "定番のカーネル", list(IM.KERNELS),
            format_func=lambda k: IM.KERNELS[k].label, index=1, key=f"{KEY}_kernel",
        )
        st.caption(IM.KERNELS[kernel_key].note)

        edit = st.toggle(
            "自分で数値を書き換える", value=False, key=f"{KEY}_edit",
            help="9 つの数値を直接いじって、何が起きるか試せます。",
        )

    base_kernel = IM.KERNELS[kernel_key].as_array()
    if edit:
        edited = st.data_editor(
            pd.DataFrame(base_kernel, columns=["左", "中", "右"], index=["上", "中", "下"]),
            width="stretch", key=f"{KEY}_editor",
        )
        kernel = np.asarray(edited.to_numpy(), dtype=float)
    else:
        kernel = base_kernel

    with col_kernel:
        st.plotly_chart(
            P.kernel_figure(kernel, height=230), width="stretch", key=f"{KEY}_kernelfig"
        )

    stats = IM.kernel_stats(kernel)
    kpi_row(
        [
            Kpi("係数の合計", f"{stats['合計']:+.3f}",
                sub="1 なら明るさ保存、0 なら差分検出",
                color=theme.GOOD if abs(stats["合計"] - 1) < 1e-6
                else theme.PURPLE if abs(stats["合計"]) < 1e-6 else theme.WARN),
            Kpi("絶対値の合計", f"{stats['絶対値の合計']:.3f}", sub="効果の強さの目安"),
            Kpi("最小 / 最大", f"{stats['最小']:g} / {stats['最大']:g}",
                sub="負の値があると差分を取ります"),
        ],
        accent,
    )

    if abs(stats["合計"]) < 1e-6:
        note(
            "係数の合計が 0 です — 平坦な部分は 0 になり、変化のある場所だけが残ります。"
            "これが輪郭抽出の原理です。",
            tone="good",
        )
    elif abs(stats["合計"] - 1.0) > 0.01:
        note(
            f"係数の合計が {stats['合計']:.2f} です — 画像全体が"
            f"{'明るく' if stats['合計'] > 1 else '暗く'}なります。"
            "明るさを保ちたいなら合計を 1 にしてください。",
            tone="warn",
        )

    @st.cache_data(show_spinner="畳み込み中…")
    def convolve_cached(image, kernel):
        return IM.apply_kernel(image, kernel)

    convolved = convolve_cached(image, kernel)
    displayed = IM.normalise_for_display(convolved) if convolved.ndim == 2 else np.clip(convolved, 0, 1)

    st.plotly_chart(
        P.before_after_figure(image, displayed, ("元の画像", f"{IM.KERNELS[kernel_key].label} を適用"), height=380),
        width="stretch", key=f"{KEY}_conv",
    )
    st.caption(
        "この操作を何十層も重ね、**カーネルの中身をデータから学習させる**のが"
        "畳み込みニューラルネットワーク（CNN）です。"
        "ここで手で組んでいる「輪郭を拾うカーネル」に近いものを、CNN は自分で見つけます。"
    )

    panel("エッジ検出", "輪郭を取り出す定番の手法を比べます")

    col_method, col_sigma = st.columns([1, 1])
    edge_method = col_method.selectbox(
        "手法", list(IM.EDGE_METHODS),
        format_func=lambda k: IM.EDGE_METHODS[k], key=f"{KEY}_edge",
    )
    sigma = col_sigma.slider(
        "前処理のぼかし具合 σ", 0.5, 5.0, 2.0, 0.5, key=f"{KEY}_sigma",
        help="ぼかしてから輪郭を取ると、細かいノイズを拾わずに済みます。",
    )

    @st.cache_data(show_spinner=False)
    def edges_cached(image, method, sigma):
        return IM.detect_edges(image, method, sigma)

    edges = edges_cached(image, edge_method, float(sigma))
    binary, threshold = IM.threshold_image(image)

    col_edge, col_binary = st.columns(2)
    with col_edge:
        st.markdown(f"**{IM.EDGE_METHODS[edge_method]}**")
        st.plotly_chart(
            P.image_figure(IM.normalise_for_display(edges), height=330),
            width="stretch", key=f"{KEY}_edges",
        )
    with col_binary:
        st.markdown(f"**二値化（大津の方法・しきい値 {threshold:.3f}）**")
        st.plotly_chart(
            P.image_figure(binary, height=330), width="stretch", key=f"{KEY}_binary"
        )
    st.caption(
        "ソーベルは勾配の大きさをそのまま出すので太い線になります。"
        "キャニーは細線化と二値化まで行うので、輪郭が 1 画素の線になります。"
        "σ を上げると細かい模様を拾わなくなる代わりに、細い輪郭も消えます。"
    )

# ======================================================================
# タブ 3: 特徴量と分類
# ======================================================================
with tabs[2]:
    panel("HOG — 「どの向きの輪郭がどこにあるか」", "深層学習以前の画像認識の主力でした")

    col_orient, col_cell = st.columns(2)
    orientations = col_orient.slider("向きを何段階に分けるか", 4, 16, 9, 1, key=f"{KEY}_orient")
    cell_size = col_cell.select_slider(
        "1 区画の大きさ（画素）", options=[8, 12, 16, 24, 32], value=16, key=f"{KEY}_cell",
        help="小さくすると細かく見ますが、特徴量の次元数が急増します。",
    )

    @st.cache_data(show_spinner="HOG を計算中…")
    def hog_cached(image, orientations, cell_size):
        return IM.compute_hog(image, orientations, cell_size, 2)

    hog = hog_cached(image, int(orientations), int(cell_size))

    kpi_row(
        [
            Kpi("元の画素数", f"{int(np.prod(image.shape[:2])):,}", sub="生のまま使った場合の次元数"),
            Kpi("HOG の次元数", f"{hog.n_features:,}",
                sub="向き × 区画の数", color=theme.PURPLE),
            Kpi("圧縮率", f"{hog.n_features / np.prod(image.shape[:2]):.2f}", unit="倍",
                sub="1 未満なら情報を圧縮できています",
                color=score_color(hog.n_features / np.prod(image.shape[:2]), good=0.5, bad=2.0)),
        ],
        accent,
    )

    st.plotly_chart(
        P.before_after_figure(image, hog.visualization, ("元の画像", "HOG の可視化"), height=380),
        width="stretch", key=f"{KEY}_hog",
    )
    st.caption(
        "右の図の小さな星は、その区画で「どの向きの輪郭がどれだけ強いか」を表しています。"
        "人間の目には元画像より情報が減って見えますが、"
        "**モデルにとっては大小関係が明確で扱いやすい形**になっています。"
    )

    panel("特徴量を変えて数字を分類する", "手書き数字 8×8 を SVM で当てます")

    st.caption(
        "ラボ 5（次元削減）と同じデータです。特徴量の作り方を変えると、"
        "同じモデル・同じデータでも正解率が変わります。"
    )

    @st.cache_data(show_spinner="3 通りの特徴量で学習中…")
    def compare_features():
        images, targets = IM.digit_images()
        rows = []
        results = {}
        for key, label in IM.FEATURE_METHODS.items():
            features = IM.digit_features(images, key)
            result = IM.classify_digits(features, targets, n_splits=5, seed=0)
            rows.append(
                {
                    "特徴量": label,
                    "次元数": result["n_features"],
                    "正解率": round(result["accuracy"], 4),
                }
            )
            results[key] = (result, targets)
        return pd.DataFrame(rows).sort_values("正解率", ascending=False), results

    comparison, results = compare_features()
    digit_scores = {
        IM.FEATURE_METHODS[key]: float(result["accuracy"])
        for key, (result, _) in results.items()
    }

    col_table, col_cm = st.columns([1, 1.1])
    with col_table:
        st.markdown("**特徴量ごとの成績**")
        st.dataframe(comparison, width="stretch", hide_index=True)
        st.caption(
            "8×8 という小さな画像では、生のピクセル値でも十分に高い精度が出ます。"
            "HOG が本領を発揮するのは、もっと大きく複雑な画像です。"
            "**「高度な特徴量が常に勝つ」わけではない**ことも、ここで確かめられます。"
        )
    with col_cm:
        shown = st.selectbox(
            "混同行列を見る特徴量", list(IM.FEATURE_METHODS),
            format_func=lambda k: IM.FEATURE_METHODS[k], key=f"{KEY}_cmfeat",
        )
        result, targets = results[shown]
        st.plotly_chart(
            TabP.confusion_figure(targets, result["predicted"], [str(i) for i in range(10)]),
            width="stretch", key=f"{KEY}_digitcm",
        )

# ======================================================================
# タブ 4: 信号と周波数
# ======================================================================
with tabs[3]:
    sidebar_section("信号")
    signal_kind = st.sidebar.radio(
        "波の種類", ["合成波", "チャープ（周波数が変化）"],
        key=f"{KEY}_sigkind",
        help="チャープは時間とともに音が高くなる波。FFT の限界が分かります。",
    )
    sample_rate = st.sidebar.select_slider(
        "標本化周波数 (Hz)", options=[100, 200, 500, 1000], value=500,
        key=f"{KEY}_sr", help="1 秒あたり何回測るか。この半分までしか観測できません。",
    )
    noise_level = st.sidebar.slider(
        "ノイズの強さ", 0.0, 1.5, 0.0, 0.05, key=f"{KEY}_noise"
    )

    if signal_kind == "合成波":
        panel("波を混ぜる", "3 つの正弦波を足し合わせます")
        cols = st.columns(3)
        components = []
        defaults = [(5.0, 1.0), (20.0, 0.5), (60.0, 0.25)]
        for i, (default_hz, default_amp) in enumerate(defaults):
            with cols[i]:
                hz = st.slider(
                    f"成分 {i + 1} の周波数 (Hz)", 1.0, float(sample_rate) / 2 - 1,
                    min(default_hz, sample_rate / 2 - 1), 1.0, key=f"{KEY}_hz{i}",
                )
                amp = st.slider(
                    f"成分 {i + 1} の振幅", 0.0, 1.0, default_amp, 0.05, key=f"{KEY}_amp{i}"
                )
                components.append(SG.Component(float(hz), float(amp)))
        config = SG.SignalConfig(
            duration=2.0, sample_rate=int(sample_rate),
            components=tuple(components), noise=float(noise_level), seed=0,
        )
        time, values = SG.make_signal(config)
    else:
        panel("チャープ", "時間とともに周波数が上がっていく波")
        config = SG.SignalConfig(
            duration=3.0, sample_rate=int(sample_rate),
            components=(), noise=float(noise_level), seed=0,
        )
        time, values = SG.make_chirp(config, 2.0, min(80.0, sample_rate / 2 - 5))

    sidebar_section("フィルタ")
    filter_kind = st.sidebar.selectbox(
        "種類", list(SG.FILTER_TYPES),
        format_func=lambda k: SG.FILTER_TYPES[k], key=f"{KEY}_filter",
    )
    low_hz = st.sidebar.slider(
        "しきい値の周波数 (Hz)", 1.0, float(sample_rate) / 2 - 1, 15.0, 1.0,
        key=f"{KEY}_lowhz", disabled=filter_kind == "none",
    )
    high_hz = st.sidebar.slider(
        "上限の周波数 (Hz)", 1.0, float(sample_rate) / 2 - 1, 40.0, 1.0,
        key=f"{KEY}_highhz", disabled=filter_kind != "bandpass",
    )

    filtered = SG.apply_filter(
        values, int(sample_rate), filter_kind, float(low_hz), float(high_hz)
    )
    original_spectrum = SG.spectrum(values, int(sample_rate))
    filtered_spectrum = SG.spectrum(filtered, int(sample_rate))
    peaks = filtered_spectrum.peaks(5)

    kpi_row(
        [
            Kpi("点数", f"{len(values):,}", sub=f"{config.duration:.0f} 秒 × {sample_rate} Hz"),
            Kpi("観測できる上限", f"{original_spectrum.nyquist:.0f}", unit="Hz",
                sub="ナイキスト周波数（標本化の半分）", color=theme.ORANGE),
            Kpi("検出した山", f"{len(peaks)}", sub="スペクトルの目立つピーク",
                color=theme.PURPLE),
            Kpi("最も強い成分",
                f"{peaks[0][0]:.1f} Hz" if peaks else "—",
                sub=f"振幅 {peaks[0][1]:.3f}" if peaks else "山が見つかりません"),
        ],
        accent,
    )

    panel("波形", "時間の並びを見ているだけでは、何が混ざっているか分かりません")
    series = [("元の信号", values, theme.rgba(theme.CYAN, 0.65))]
    if filter_kind != "none":
        series.append(("フィルタ後", filtered, theme.PINK))
    st.plotly_chart(
        P.waveform_figure(time, series), width="stretch", key=f"{KEY}_wave"
    )

    panel("周波数に分解する（FFT）", "どの高さの成分がどれだけ含まれているか")
    st.plotly_chart(
        P.spectrum_figure(
            filtered_spectrum.frequencies, filtered_spectrum.amplitudes, peaks,
            max_hz=original_spectrum.nyquist,
        ),
        width="stretch", key=f"{KEY}_spectrum",
    )
    if signal_kind == "合成波":
        st.caption(
            "設定した周波数のところに、設定した振幅どおりの山が立ちます。"
            "ぐちゃぐちゃに見えた波形が、**実は数本の正弦波の足し算だった**ことが分かります。"
            "ノイズを上げると、山の間が持ち上がってくるのが見えます。"
        )
    else:
        st.caption(
            "チャープでは山が 1 本に立たず、広い範囲に散らばります。"
            "**FFT は「全体で何が鳴っていたか」しか答えられず、"
            "「いつ鳴ったか」は分かりません。** それを見るのが次のスペクトログラムです。"
        )

    panel("スペクトログラム", "短い窓をずらしながら FFT をかけ、時間 × 周波数で見ます")
    window = st.select_slider(
        "窓の長さ（点数）", options=[32, 64, 128, 256], value=128, key=f"{KEY}_window",
        help="短くすると時間の分解能が上がり、長くすると周波数の分解能が上がります。",
    )
    frequencies, times, decibels = SG.spectrogram(filtered, int(sample_rate), int(window))
    st.plotly_chart(
        P.spectrogram_figure(frequencies, times, decibels),
        width="stretch", key=f"{KEY}_spectrogram",
    )
    st.caption(
        "横が時間、縦が周波数、色が強さです。チャープを選ぶと、"
        "明るい帯が右上がりに伸びていく — つまり音が高くなっていくのが見えます。"
        "**窓を短くすると時間に強く、長くすると周波数に強くなります**（両立はできません）。"
    )

    panel("標本化が粗いとどうなるか", "エイリアシング — 速い波が遅い波に化ける現象")
    col_true, col_rate = st.columns(2)
    true_hz = col_true.slider("本当の周波数 (Hz)", 5.0, 100.0, 40.0, 1.0, key=f"{KEY}_truehz")
    demo_rate = col_rate.select_slider(
        "標本化周波数 (Hz)", options=[20, 30, 50, 80, 100, 200, 500], value=50,
        key=f"{KEY}_demorate",
    )
    dense_t, dense_v, sample_t, sample_v, apparent = SG.aliasing_demo(
        float(true_hz), int(demo_rate), duration=0.5
    )
    st.plotly_chart(
        P.aliasing_figure(dense_t, dense_v, sample_t, sample_v, apparent),
        width="stretch", key=f"{KEY}_alias",
    )
    if apparent < true_hz - 0.01:
        note(
            f"本当は {true_hz:.0f} Hz なのに、{apparent:.1f} Hz に見えています — "
            f"標本化周波数 {demo_rate} Hz では {demo_rate / 2:.0f} Hz までしか正しく観測できません。",
            tone="bad",
        )
    else:
        note(
            f"標本化周波数 {demo_rate} Hz なら {demo_rate / 2:.0f} Hz まで観測できるので、"
            f"{true_hz:.0f} Hz は正しく捉えられています。",
            tone="good",
        )
    st.caption(
        "**測りたい周波数の 2 倍より速く測る**必要があります（標本化定理）。"
        "音声を 44.1kHz で記録するのは、人間の可聴上限 20kHz の 2 倍を超えるためです。"
    )

# ======================================================================
# 実験の記録
# ======================================================================

EL.record_panel(
    lab="画像・信号",
    params={
        "画像": IM.IMAGES[image_key].label,
        "グレースケール": bool(use_gray or not has_color),
        "カーネル": IM.KERNELS[kernel_key].label,
        "エッジ検出": IM.EDGE_METHODS[edge_method],
        "ぼかしσ": float(sigma),
        "HOGの向き数": int(orientations),
        "HOGの区画": int(cell_size),
    },
    metrics={
        "HOGの次元数": float(hog.n_features),
        "カーネル係数の合計": float(stats["合計"]),
        **{f"数字分類:{name}": score for name, score in digit_scores.items()},
    },
    key=KEY,
    default_experiment=f"画像/{image_key}",
)

# 解説の数式に、いまのぼかし具合と標本化周波数を差し込む。
explain(
    "imaging",
    values={
        "sigma": float(sigma),
        "fs": int(sample_rate),
        "nyquist": sample_rate / 2,
    },
)
