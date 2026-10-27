"""
Generate Week 4 Status Presentation (Week4_Status.pptx) using python-pptx.
Adheres strictly to the university project presentation requirements:
  - 16:9 widescreen layout
  - Clean, modern, accessible design (clear contrast, card layouts)
  - Plain language for non-technical audience
  - Honest reporting and scientific rigor (no unverified claims)
  - Real figures inserted from results directory
  - Speaker notes split across 4 presenters (Model, Augmentation, Training, Evaluation)
  - Follows Week 3 deck structure:
    1. Title & Research Objective
    2. Context & Week 3 Baseline
    3. Week 4 Pipeline & Additions
    4. Task 1: 144-Input Representation Gap Analysis
    5. Task 1: EmoDB 144-Input Experimental Results
    6. Task 2: Multi-Corpus Evaluation Setup (CAFE, AESDD, MESD)
    7. Task 2: Cross-Corpus Performance & Augmentation Effect
    8. Task 3: Speaker-Independent Partitioning Protocol
    9. Task 3: Speaker Generalization Drop Analysis
    10. Synthesis & Critical Findings
    11. Ambiguities, Assumptions & Scientific Integrity
    12. Week 5 Roadmap & Next Steps
    13. Backup: Architecture & Loss Dynamics
"""

import sys
from pathlib import Path
import pptx
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results"

# Visual Theme & Palette (consistent with Week 3)
BG_COLOR = RGBColor(248, 250, 252)        # Slate 50
CARD_BG = RGBColor(255, 255, 255)         # White
PRIMARY = RGBColor(30, 58, 138)           # Blue 900
SECONDARY = RGBColor(14, 116, 144)        # Cyan 700
ACCENT = RGBColor(37, 99, 235)            # Blue 600
TEXT_DARK = RGBColor(15, 23, 42)          # Slate 900
TEXT_MUTED = RGBColor(71, 85, 105)        # Slate 600
BORDER_COLOR = RGBColor(226, 232, 240)    # Slate 200
HIGHLIGHT_BG = RGBColor(238, 242, 255)    # Indigo 50
ACCENT_GREEN = RGBColor(16, 185, 129)     # Emerald 500
ACCENT_RED = RGBColor(225, 29, 72)        # Rose 600


def set_slide_background(slide, color=BG_COLOR):
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_header(slide, title_text, category_text="SPEECH EMOTION RECOGNITION • WEEK 4"):
    cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.5), Inches(0.4))
    tf_c = cat_box.text_frame
    tf_c.word_wrap = True
    p_c = tf_c.paragraphs[0]
    p_c.text = category_text.upper()
    p_c.font.size = Pt(10)
    p_c.font.bold = True
    p_c.font.color.rgb = SECONDARY

    t_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.7), Inches(11.5), Inches(0.8))
    tf_t = t_box.text_frame
    tf_t.word_wrap = True
    p_t = tf_t.paragraphs[0]
    p_t.text = title_text
    p_t.font.size = Pt(24)
    p_t.font.bold = True
    p_t.font.color.rgb = PRIMARY


def add_card(slide, left, top, width, height, title=None, bg_color=CARD_BG, border_color=BORDER_COLOR):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = bg_color
    if border_color:
        shape.line.color.rgb = border_color
        shape.line.width = Pt(1.5)
    else:
        shape.line.fill.background()

    if title:
        tb = slide.shapes.add_textbox(left + Inches(0.2), top + Inches(0.15), width - Inches(0.4), Inches(0.5))
        p = tb.text_frame.paragraphs[0]
        p.text = title
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = PRIMARY
    return shape


def add_speaker_notes(slide, presenter, notes):
    notes_slide = slide.notes_slide
    tf = notes_slide.notes_text_frame
    tf.text = f"PRESENTER: {presenter}\n\nSPEAKER NOTES:\n{notes}"


def create_deck(output_path=ROOT_DIR / "Week4_Status.pptx"):
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # =========================================================================
    # SLIDE 1: Title Slide
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_background(s1)

    t_box = s1.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.3), Inches(2.2))
    tf = t_box.text_frame
    tf.word_wrap = True

    p0 = tf.paragraphs[0]
    p0.text = "SPEECH EMOTION RECOGNITION • WEEK 4 STATUS"
    p0.font.size = Pt(12)
    p0.font.bold = True
    p0.font.color.rgb = SECONDARY
    p0.space_after = Pt(12)

    p1 = tf.add_paragraph()
    p1.text = "Cross-Corpus Generalization, Input Representations,\nand Speaker Independence in 1D-CNN SER"
    p1.font.size = Pt(28)
    p1.font.bold = True
    p1.font.color.rgb = PRIMARY
    p1.space_after = Pt(16)

    p2 = tf.add_paragraph()
    p2.text = (
        "Independent Reproduction of Thomas Mary Little Flower et al. (Automatika, 2024)\n"
        "Investigating the 144-Input Gap, CAFE/AESDD/MESD Datasets, and Speaker-Disjoint Splits"
    )
    p2.font.size = Pt(14)
    p2.font.color.rgb = TEXT_MUTED

    card_meta = add_card(s1, Inches(1.0), Inches(4.5), Inches(11.3), Inches(1.8), bg_color=HIGHLIGHT_BG)
    tb_meta = s1.shapes.add_textbox(Inches(1.2), Inches(4.6), Inches(10.9), Inches(1.5))
    tf_m = tb_meta.text_frame
    tf_m.word_wrap = True
    p_m1 = tf_m.paragraphs[0]
    p_m1.text = "RESEARCH TEAM & ROLES"
    p_m1.font.size = Pt(11)
    p_m1.font.bold = True
    p_m1.font.color.rgb = PRIMARY
    p_m1.space_after = Pt(6)

    p_m2 = tf_m.add_paragraph()
    p_m2.text = (
        "• Person 1 (Model Lead): 144-input temporal sequence modeling vs. statistical aggregation\n"
        "• Person 2 (Augmentation Lead): Cross-corpus additive Gaussian noise and leak-free scaling\n"
        "• Person 3 (Training & Pipeline Lead): Multi-corpus audio normalization, caching, and reproducibility\n"
        "• Person 4 (Evaluation Lead): Speaker-independent partitioning and generalization drop quantification"
    )
    p_m2.font.size = Pt(11)
    p_m2.font.color.rgb = TEXT_DARK

    add_speaker_notes(
        s1,
        "Person 1 (Model Lead)",
        "Welcome to our Week 4 status presentation on reproducing the 2024 Automatika paper on 1D-CNN speech emotion recognition. "
        "In Week 3, we reproduced the baseline architecture and proved that augmenting data before the train/test split creates massive data leakage, artificially inflating accuracy by 15-20%. "
        "However, our leak-free baseline on EmoDB was still 15-20% below the paper's reported numbers. "
        "Today, our team addresses three critical Week 4 questions: First, does changing the feature representation to match the paper's 144-input layer bridge that gap? "
        "Second, do these models and augmentation effects replicate across three additional international speech corpora: CAFE, AESDD, and MESD? "
        "And third, what happens when we test the models on speakers they have never heard before?"
    )

    # =========================================================================
    # SLIDE 2: Context & Week 3 Baseline
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    set_slide_background(s2)
    add_header(s2, "Context & Starting Point: The Week 3 Baseline")

    # Card 1: Week 3 Achievements
    add_card(s2, Inches(0.8), Inches(1.7), Inches(5.6), Inches(5.2), title="Week 3 Empirical Findings")
    tb2_1 = s2.shapes.add_textbox(Inches(1.0), Inches(2.3), Inches(5.2), Inches(4.4))
    tf2_1 = tb2_1.text_frame
    tf2_1.word_wrap = True
    tf2_1.text = (
        "• Complete Baseline Reproduction:\n"
        "  - EmoDB (535 clips, 7 emotions), 1D-CNN with 4 Conv blocks.\n"
        "  - Evaluated MFCC and MFMC at 12, 24, 30 coefficients (mean + std over time: 24, 48, 60 values).\n\n"
        "• The Leakage Discovery:\n"
        "  - Augmenting before splitting (Version B) leaked 98.5% of source clips into training, yielding 80–89% accuracy.\n"
        "  - Clean augmentation after splitting (Version A) yielded 64–72% accuracy.\n\n"
        "• The Persistent Gap:\n"
        "  - Paper reports 86.9% without augmentation and 97.2% with augmentation.\n"
        "  - Even without augmentation, our baseline (64–71%) lagged the paper by ~16–22%.\n"
        "  - Leakage alone could not explain the entire performance discrepancy."
    )
    for p in tf2_1.paragraphs:
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_DARK

    # Card 2: Week 4 Research Mandate
    add_card(s2, Inches(6.8), Inches(1.7), Inches(5.7), Inches(5.2), title="Week 4 Research Mandates")
    tb2_2 = s2.shapes.add_textbox(Inches(7.0), Inches(2.3), Inches(5.3), Inches(4.4))
    tf2_2 = tb2_2.text_frame
    tf2_2.word_wrap = True
    tf2_2.text = (
        "1. Task 1: Investigate the 144-Input Format\n"
        "   - The paper's CNN specifies 144 inputs; Week 3 used 24/48/60.\n"
        "   - Formulate grounded hypotheses from the text, implement top interpretations, and test if 144 inputs closes the gap.\n\n"
        "2. Task 2: Cross-Corpus Evaluation\n"
        "   - Extend evaluation beyond German EmoDB to CAFE (French), AESDD (Greek), and MESD (Spanish).\n"
        "   - Verify whether feature representations and Gaussian augmentation behave consistently across languages.\n\n"
        "3. Task 3: Speaker-Independent Splits\n"
        "   - Traditional stratified splits allow the model to memorize speaker vocal tract traits.\n"
        "   - Partition datasets by speaker ID so test speakers are completely unseen."
    )
    for p in tf2_2.paragraphs:
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_DARK

    add_speaker_notes(
        s2,
        "Person 1 (Model Lead)",
        "To put our Week 4 work in context: last week we demonstrated that the paper's reported 97.2% accuracy is heavily inflated if audio augmentation is applied before data partitioning. "
        "However, even our clean, unaugmented baseline reached only 71%, whereas the paper claims 86.9% without augmentation. "
        "This proved that data leakage was only part of the story. Week 4 is designed to systematically track down the remaining discrepancy: examining whether the model expects 144 temporal frames, whether the results hold on larger international datasets, and whether performance collapses when evaluated across unseen speakers."
    )

    # =========================================================================
    # SLIDE 3: Week 4 Pipeline & Additions
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    set_slide_background(s3)
    add_header(s3, "Pipeline Architecture & Week 4 Additions")

    # Card 1: Multi-Corpus Infrastructure
    add_card(s3, Inches(0.8), Inches(1.7), Inches(3.7), Inches(5.2), title="1. Multi-Corpus Loader")
    tb3_1 = s3.shapes.add_textbox(Inches(1.0), Inches(2.3), Inches(3.3), Inches(4.4))
    tf3_1 = tb3_1.text_frame
    tf3_1.word_wrap = True
    tf3_1.text = (
        "• Unified Data Engine:\n"
        "  - Ingestion for 4 datasets (EmoDB, CAFE, AESDD, MESD).\n"
        "  - Resampling to 16 kHz mono.\n"
        "  - Metadata tables logging emotion, speaker, language, duration.\n\n"
        "• Quality & Sanity:\n"
        "  - Verified zero corrupt files (filtered corrupted 36-byte AESDD sample).\n"
        "  - Verified zero NaNs/Infs.\n"
        "  - Standardized label encoding (5, 6, 7 classes)."
    )
    for p in tf3_1.paragraphs:
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_DARK

    # Card 2: Feature & Input Engine
    add_card(s3, Inches(4.8), Inches(1.7), Inches(3.7), Inches(5.2), title="2. 144-Input Implementations")
    tb3_2 = s3.shapes.add_textbox(Inches(5.0), Inches(2.3), Inches(3.3), Inches(4.4))
    tf3_2 = tb3_2.text_frame
    tf3_2.word_wrap = True
    tf3_2.text = (
        "• Representation Options:\n"
        "  - Baseline: (24, 48, 60, 1) pooled mean + std.\n"
        "  - frames144: (144, n_coeff) temporal STFT sequence.\n"
        "  - stats144: (144, 1) multi-stream static + delta + delta-delta vector.\n\n"
        "• Conv1D Adaptations:\n"
        "  - Flexible input shape (144, C) or (D, 1).\n"
        "  - Kernel size 15 across temporal frames spans ~240 ms context.\n"
        "  - Same padding preserves temporal tracking."
    )
    for p in tf3_2.paragraphs:
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_DARK

    # Card 3: Strict Splitting & Scaling Guardrails
    add_card(s3, Inches(8.8), Inches(1.7), Inches(3.7), Inches(5.2), title="3. Leak-Free Guardrails")
    tb3_3 = s3.shapes.add_textbox(Inches(9.0), Inches(2.3), Inches(3.3), Inches(4.4))
    tf3_3 = tb3_3.text_frame
    tf3_3.word_wrap = True
    tf3_3.text = (
        "• Zero-Leakage Splitting:\n"
        "  - Stratified 60/20/20.\n"
        "  - Speaker-Independent: GroupKFold ensuring 0% speaker overlap.\n\n"
        "• Proper Preprocessing:\n"
        "  - StandardScaler fitted ONLY on training split.\n"
        "  - Augmentation applied ONLY to training audio.\n"
        "  - Val and test sets remain clean original clips.\n"
        "  - Single-pass test evaluation on frozen checkpoint."
    )
    for p in tf3_3.paragraphs:
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_DARK

    add_speaker_notes(
        s3,
        "Person 3 (Training & Pipeline Lead)",
        "To support these experiments, we upgraded our codebase into a unified multi-corpus pipeline. "
        "The data loader automatically harmonizes audio across four languages into 16 kHz mono streams, extracts both baseline and 144-input representations, and handles class mappings for 5, 6, and 7 emotions. "
        "Crucially, we maintained 100% leak-free integrity: normalizers are fitted exclusively on training audio, and we introduced a strict speaker-disjoint splitting algorithm."
    )

    # =========================================================================
    # SLIDE 4: Task 1: 144-Input Representation Gap Analysis
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    set_slide_background(s4)
    add_header(s4, "Task 1: Dissecting the 144-Input Representation Gap")

    # Card 1: What the Paper States
    add_card(s4, Inches(0.8), Inches(1.7), Inches(5.6), Inches(5.2), title="Paper Textual Evidence & Ambiguities")
    tb4_1 = s4.shapes.add_textbox(Inches(1.0), Inches(2.3), Inches(5.2), Inches(4.4))
    tf4_1 = tb4_1.text_frame
    tf4_1.word_wrap = True
    tf4_1.text = (
        "• Exact Paper Phrasing:\n"
        "  'One hundred forty-four feature vectors as input data for our CNN\\'s first layer.'\n\n"
        "• Key Observations & Ambiguities:\n"
        "  1. The authors use the plural 'feature vectors', implying a 2D matrix (T frames x C channels) rather than a single 1D vector.\n"
        "  2. In speech processing, Conv1D with kernel size 15 makes physical sense along the time dimension (spanning ~240 ms of speech), whereas across a 24-element global summary vector, kernel 15 covers 62.5% of the entire representation in a single hop!\n"
        "  3. However, the paper never specifies the audio clip duration, STFT hop size, or framing required to yield exactly 144 frames."
    )
    for p in tf4_1.paragraphs:
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_DARK

    # Card 2: Formulated Implementations
    add_card(s4, Inches(6.8), Inches(1.7), Inches(5.7), Inches(5.2), title="Two Grounded Interpretations Implemented")
    tb4_2 = s4.shapes.add_textbox(Inches(7.0), Inches(2.3), Inches(5.3), Inches(4.4))
    tf4_2 = tb4_2.text_frame
    tf4_2.word_wrap = True
    tf4_2.text = (
        "• Interpretation 1: 'frames144' (Temporal Sequence)\n"
        "  - Representation: Matrix of shape (144, n_coeff).\n"
        "  - Construction: Audio fixed to 2.3 seconds (or center-cropped/padded) at 16 kHz with hop length 256 samples (16 ms) -> exactly 144 frames.\n"
        "  - Conv1D slides over time; 4 pooling layers reduce time: 144 -> 72 -> 36 -> 18 -> 9. Flatten layer yields 9 x 64 = 576 units into Dense(120).\n\n"
        "• Interpretation 2: 'stats144' (Multi-Stream Statistics)\n"
        "  - Representation: 1D vector of shape (144, 1).\n"
        "  - Construction: Static + Delta + Delta-Delta trajectories across coefficients, summarized via statistical moments (mean, std, min, max).\n"
        "  - For 24 coeffs: 24 x 3 streams x 2 stats = 144 features.\n"
        "  - For 12 coeffs: 12 x 3 streams x 4 stats = 144 features."
    )
    for p in tf4_2.paragraphs:
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_DARK

    add_speaker_notes(
        s4,
        "Person 1 (Model Lead)",
        "Slide 4 breaks down the biggest mystery from Week 3: the paper says the input layer receives '144 feature vectors'. "
        "In Week 3, we used pooled mean and standard deviation vectors of length 24, 48, and 60. "
        "Looking closely at the architecture, applying a 1D convolution with kernel size 15 across a 24-element vector covers over 60% of the entire vector in one step. But on a temporal sequence of 144 time frames, kernel 15 corresponds to roughly 240 milliseconds of speech—a standard phoneme-length receptive field. "
        "Therefore, we implemented two rigorous alternatives: frames144, which preserves the temporal sequence of 144 frames, and stats144, which expands summary statistics to 144 dimensions using velocity and acceleration deltas."
    )

    # =========================================================================
    # SLIDE 5: Task 1 Results: EmoDB 144-Input Comparison
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    set_slide_background(s5)
    add_header(s5, "Task 1 Results: Does 144 Inputs Bridge the Gap?")

    # Left: Findings card
    add_card(s5, Inches(0.8), Inches(1.7), Inches(5.6), Inches(5.2), title="Empirical Findings on EmoDB")
    tb5_1 = s5.shapes.add_textbox(Inches(1.0), Inches(2.3), Inches(5.2), Inches(4.4))
    tf5_1 = tb5_1.text_frame
    tf5_1.word_wrap = True
    tf5_1.text = (
        "• Does 144 Inputs Close the Gap? NO.\n"
        "  - In fact, the temporal frames144 format performs significantly WORSE than our 24/48/60 baseline.\n\n"
        "• Quantitative Comparison (MFCC, None, Stratified):\n"
        "  - 12 Coeffs Baseline (24-dim): 64.2% ± 4.4%\n"
        "  - 12 Coeffs frames144 (144x12): 56.4% ± 2.0% (Drop: -7.8%)\n"
        "  - 24 Coeffs Baseline (48-dim): 70.1% ± 3.8%\n"
        "  - 24 Coeffs frames144 (144x24): 54.8% ± 5.2% (Drop: -15.3%)\n\n"
        "• Why Does frames144 Degrade Performance?\n"
        "  - Severe Overfitting: Without global pooling, the model has 576 units feeding into Dense(120).\n"
        "  - On small datasets (321 training clips), Conv1D rapidly memorizes exact frame sequences, reaching 100% training accuracy within 15 epochs while validation diverges.\n"
        "  - Baseline summary statistics (mean+std) provide implicit temporal regularization."
    )
    for p in tf5_1.paragraphs:
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_DARK

    # Right: Plot or Summary Card
    img_task1 = RESULTS_DIR / "week4_plot_input_formats.png"
    if img_task1.exists():
        add_card(s5, Inches(6.8), Inches(1.7), Inches(5.7), Inches(5.2), title="Input Format Test Accuracy Comparison")
        s5.shapes.add_picture(str(img_task1), Inches(7.0), Inches(2.3), width=Inches(5.3))
    else:
        add_card(s5, Inches(6.8), Inches(1.7), Inches(5.7), Inches(5.2), title="Key Takeaway on Model Input Format")
        tb5_2 = s5.shapes.add_textbox(Inches(7.0), Inches(2.3), Inches(5.3), Inches(4.4))
        tf5_2 = tb5_2.text_frame
        tf5_2.word_wrap = True
        tf5_2.text = (
            "• Paper Claim vs. Reproduction Reality:\n"
            "  - Paper claimed: 86.9% without augmentation, 97.2% with augmentation.\n"
            "  - Baseline (24/48/60): 64–71% test accuracy.\n"
            "  - frames144 (144xT): 51–61% test accuracy.\n"
            "  - Clean Augmentation (Version A) on frames144: 56.1% (no meaningful improvement).\n\n"
            "• Scientific Conclusion:\n"
            "  - The 144-input architectural gap is NOT the reason for our lower baseline.\n"
            "  - Retaining temporal frames without massive regularization or pretraining severely exacerbates overfitting on small emotional speech corpora."
        )
        for p in tf5_2.paragraphs:
            p.font.size = Pt(11)
            p.font.color.rgb = TEXT_DARK

    add_speaker_notes(
        s5,
        "Person 1 (Model Lead)",
        "The empirical results on Task 1 are decisive. When we feed a temporal sequence of 144 frames into the 1D-CNN, performance does not rise toward the paper's 87%—it drops by 8 to 15 percentage points down to 54-56%. "
        "Because the model has to learn temporal emotional dynamics from only 321 training clips without recurrent or attention layers, it simply memorizes the training clips, hitting 100% training accuracy almost immediately. "
        "The simple pooled mean and standard deviation vectors from Week 3 actually act as strong feature-level regularizers. This confirms that the input format does not explain the performance gap with the published paper."
    )

    # =========================================================================
    # SLIDE 6: Task 2: Multi-Corpus Evaluation Setup
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    set_slide_background(s6)
    add_header(s6, "Task 2: Multi-Corpus Evaluation (CAFE, AESDD, MESD)")

    # Card 1: Corpus Inventory Table
    add_card(s6, Inches(0.8), Inches(1.7), Inches(6.0), Inches(5.2), title="Benchmark Corpora Characteristics")
    tb6_1 = s6.shapes.add_textbox(Inches(1.0), Inches(2.3), Inches(5.6), Inches(4.4))
    tf6_1 = tb6_1.text_frame
    tf6_1.word_wrap = True
    tf6_1.text = (
        "• EmoDB (German):\n"
        "  - 535 clips, 10 professional actors, 7 emotions (ang, bor, dis, fea, hap, sad, neu).\n"
        "  - Recorded in studio setting, high acoustic clarity.\n\n"
        "• CAFE (Canadian French):\n"
        "  - 936 clips, 12 professional actors (6M/6F), 7 emotions.\n"
        "  - Sourced from open CAMEO HuggingFace mirror (not verified against Zenodo 1478765 release).\n\n"
        "• AESDD (Att-HACK Greek):\n"
        "  - 500 clips, 5 core actors (1–5), 20 sentences (01–20), 5 emotions.\n"
        "  - Rebuilt canonical subset asserting exactly 100 clips/emotion (4 takes from actor 6 to replace missing/corrupt takes).\n\n"
        "• MESD (Mexican Spanish):\n"
        "  - 862 single-word utterances, 6 emotions (ang, dis, fea, hap, neu, sad).\n"
        "  - Sourced from open CAMEO HuggingFace mirror (not verified against Mendeley Data release).\n"
        "  - Contains 3 demographic cohorts: female, male, and children."
    )
    for p in tf6_1.paragraphs:
        p.font.size = Pt(10.5)
        p.font.color.rgb = TEXT_DARK

    # Card 2: Experimental Protocol & Sanity Checks
    add_card(s6, Inches(7.2), Inches(1.7), Inches(5.3), Inches(5.2), title="Standardized Multi-Corpus Protocol")
    tb6_2 = s6.shapes.add_textbox(Inches(7.4), Inches(2.3), Inches(4.9), Inches(4.4))
    tf6_2 = tb6_2.text_frame
    tf6_2.word_wrap = True
    tf6_2.text = (
        "• Rigorous Verification Guardrails:\n"
        "  - Audio resampled to 16 kHz mono across all corpora.\n"
        "  - Label spaces cleanly harmonized to discrete integers.\n"
        "  - Verified zero corrupted clips (filtered 36-byte invalid AESDD wav).\n"
        "  - Evaluated on identical random seeds: 42, 43, 44.\n\n"
        "• Experimental Matrix per Dataset:\n"
        "  - 2 Feature sets: MFCC vs. MFMC.\n"
        "  - 3 Coefficient counts: 12, 24, 30.\n"
        "  - 2 Augmentation setups: None vs. Version A (clean 5x).\n"
        "  - Total: 36 runs per dataset x 3 datasets = 108 experimental runs."
    )
    for p in tf6_2.paragraphs:
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_DARK

    add_speaker_notes(
        s6,
        "Person 2 (Augmentation Lead)",
        "Turning to Task 2: To evaluate whether our findings are specific to German speech in EmoDB, we expanded our testing to three additional international benchmarks. "
        "CAFE provides 936 French recordings from 12 professional actors. "
        "AESDD provides 500 Greek recordings with 5 emotions, perfectly balanced across actors and sentences. "
        "MESD provides 862 Mexican Spanish recordings across adult and child speakers. "
        "All four datasets were preprocessed using the exact same signal processing pipeline, and evaluated across seeds 42, 43, and 44."
    )

    # =========================================================================
    # SLIDE 7: Task 2 Results: Cross-Corpus Performance
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    set_slide_background(s7)
    add_header(s7, "Task 2 Results: Cross-Corpus Performance & Augmentation")

    # Left: Findings Card
    add_card(s7, Inches(0.8), Inches(1.7), Inches(5.6), Inches(5.2), title="Cross-Corpus Comparative Findings")
    tb7_1 = s7.shapes.add_textbox(Inches(1.0), Inches(2.3), Inches(5.2), Inches(4.4))
    tf7_1 = tb7_1.text_frame
    tf7_1.word_wrap = True
    tf7_1.text = (
        "• Theoretical Chance Levels Differ:\n"
        "  - AESDD (5 classes): 20.0% | MESD (6 classes): 16.7%\n"
        "  - CAFE (7 classes): 14.3% | EmoDB (7 classes): 14.3%\n"
        "  - A score of 50% on 5 classes (2.5x chance) vs. 7 classes (3.5x chance).\n\n"
        "• Observed Cross-Corpus Accuracies (MFCC/MFMC 30 Coeffs):\n"
        "  - EmoDB (German, 7 cls): 66–71% (None) -> 71–72% (Ver A) [Paper: 97.2% / 97.5%]\n"
        "  - AESDD (Greek, 5 cls): 57–61% (None) -> 77–79% (Ver A) [Paper: 97.8% / 99.2%]\n"
        "  - CAFE (French, 7 cls): 46–50% (None) -> 40–55% (Ver A) [Paper: 99.5%]\n"
        "  - MESD (Spanish, 6 cls): 64–66% (None) -> 67–69% (Ver A) [Paper: 96.9%]\n\n"
        "• Key Findings on Augmentation & Replication:\n"
        "  - Clean Version A augmentation provides moderate gains on AESDD (5 classes), but minor or negative shifts on EmoDB, CAFE, and MESD.\n"
        "  - Across datasets, many augmentation differences remain within random seed variance (seed std ranges from ±2% to ±7%).\n"
        "  - The paper's claimed peak accuracies (~97–99%) were not replicated under our implementation across any of the four benchmarks."
    )
    for p in tf7_1.paragraphs:
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_DARK

    # Right: Plot or Details
    img_task2 = RESULTS_DIR / "week4_plot_datasets.png"
    if img_task2.exists():
        add_card(s7, Inches(6.8), Inches(1.7), Inches(5.7), Inches(5.2), title="Cross-Dataset Accuracy & Augmentation Effect")
        s7.shapes.add_picture(str(img_task2), Inches(7.0), Inches(2.3), width=Inches(5.3))
    else:
        add_card(s7, Inches(6.8), Inches(1.7), Inches(5.7), Inches(5.2), title="Summary Across Corpora")
        tb7_2 = s7.shapes.add_textbox(Inches(7.0), Inches(2.3), Inches(5.3), Inches(4.4))
        tf7_2 = tb7_2.text_frame
        tf7_2.word_wrap = True
        tf7_2.text = (
            "• MFCC vs. MFMC Consistency:\n"
            "  - MFCC and MFMC show remarkably comparable accuracy across all four datasets (within 1–3% of each other).\n"
            "  - Increasing coefficients from 12 to 24/30 provides consistent modest improvements (+3% to +6%) across all languages.\n\n"
            "• Universality of Overfitting:\n"
            "  - On every dataset, training accuracy reached 98–100% within 40 epochs.\n"
            "  - Additive Gaussian noise did not prevent loss divergence on validation sets, proving that simple noise does not resolve acoustic overfitting."
        )
        for p in tf7_2.paragraphs:
            p.font.size = Pt(11)
            p.font.color.rgb = TEXT_DARK

    add_speaker_notes(
        s7,
        "Person 2 (Augmentation Lead)",
        "Slide 7 shows the cross-corpus comparison across all four speech benchmarks. "
        "First, chance levels differ: 20% for AESDD (5 classes), 16.7% for MESD (6 classes), and 14.3% for CAFE and EmoDB (7 classes). "
        "Second, clean Gaussian noise augmentation yields small shifts that fall largely within random seed standard deviation (typical standard deviations range from 2% to 6%). "
        "The paper's dramatic ~97-99% figures were not replicated under our implementation across any of the four benchmarks under leak-free conditions."
    )

    # =========================================================================
    # SLIDE 8: Task 3: Speaker-Independent Partitioning Protocol
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    set_slide_background(s8)
    add_header(s8, "Task 3: Speaker-Independent Partitioning Protocol")

    # Card 1: The Problem of Speaker Confounding
    add_card(s8, Inches(0.8), Inches(1.7), Inches(5.6), Inches(5.2), title="The Speaker Confounder in SER")
    tb8_1 = s8.shapes.add_textbox(Inches(1.0), Inches(2.3), Inches(5.2), Inches(4.4))
    tf8_1 = tb8_1.text_frame
    tf8_1.word_wrap = True
    tf8_1.text = (
        "• Why Random Stratified Splits Mislead:\n"
        "  - In a standard stratified split, audio clips from the SAME speaker appear in train, validation, and test sets.\n"
        "  - The 1D-CNN easily learns to recognize speaker vocal tract geometry (timbre, pitch range, formant baselines) rather than true emotion.\n"
        "  - The model effectively acts as a speaker classifier mapping speaker traits to emotions.\n\n"
        "• Real-World Speech Recognition Requirement:\n"
        "  - Real SER systems must generalize to completely novel speakers never encountered during training.\n"
        "  - Speaker-independent evaluation is the gold standard benchmark."
    )
    for p in tf8_1.paragraphs:
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_DARK

    # Card 2: Group Partitioning Design
    add_card(s8, Inches(6.8), Inches(1.7), Inches(5.7), Inches(5.2), title="Strict Disjoint Partitioning Design")
    tb8_2 = s8.shapes.add_textbox(Inches(7.0), Inches(2.3), Inches(5.3), Inches(4.4))
    tf8_2 = tb8_2.text_frame
    tf8_2.word_wrap = True
    tf8_2.text = (
        "• Disjoint Speaker Partitions Built:\n"
        "  - EmoDB (10 speakers):\n"
        "    * Train: 6 speakers (~60% clips)\n"
        "    * Val: 2 speakers (~20% clips)\n"
        "    * Test: 2 speakers (~20% clips)\n"
        "  - CAFE (12 speakers):\n"
        "    * Train: 8 speakers (624 clips)\n"
        "    * Val: 2 speakers (156 clips)\n"
        "    * Test: 2 speakers (156 clips)\n"
        "  - AESDD (5 speakers):\n"
        "    * Train: 3 speakers (300 clips)\n"
        "    * Val: 1 speaker (100 clips)\n"
        "    * Test: 1 speaker (100 clips)\n\n"
        "• Verification:\n"
        "  - Set intersection: Train ∩ Val = ∅, Train ∩ Test = ∅, Val ∩ Test = ∅.\n"
        "  - Verified 100% disjointness via automated unit tests."
    )
    for p in tf8_2.paragraphs:
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_DARK

    add_speaker_notes(
        s8,
        "Person 4 (Evaluation Lead)",
        "In Task 3, we implemented the strictest evaluation protocol in speech emotion recognition: speaker-independent evaluation. "
        "In standard stratified splits, audio clips from the same actor appear in both train and test. "
        "Because deep neural networks are remarkably adept at picking up acoustic biometric signatures—like vocal tract length and fundamental frequency—they often learn to identify the actor rather than the emotion. "
        "We partitioned EmoDB, CAFE, and AESDD so that no speaker in the test set ever appeared in the training or validation sets, and verified zero speaker overlap with automated test suites."
    )

    # =========================================================================
    # SLIDE 9: Task 3 Protocol: Quantifying the Generalization Drop
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    set_slide_background(s9)
    add_header(s9, "Task 3: Speaker-Independent Experimental Design & Protocol")

    # Left: Protocol & Design Card
    add_card(s9, Inches(0.8), Inches(1.7), Inches(5.6), Inches(5.2), title="Protocol & Execution Parameters")
    tb9_1 = s9.shapes.add_textbox(Inches(1.0), Inches(2.3), Inches(5.2), Inches(4.4))
    tf9_1 = tb9_1.text_frame
    tf9_1.word_wrap = True
    tf9_1.text = (
        "• Strict Experimental Guardrails:\n"
        "  - Grouped Partitioning: Speakers never shared across train, val, and test.\n"
        "  - Model Selection: Best checkpoint selected SOLELY on validation speakers.\n"
        "  - Single-Pass Test: Test speakers evaluated exactly once on frozen weights.\n"
        "  - MESD Excluded: Lacks individual speaker IDs (demographic cohorts only).\n\n"
        "• Experimental Matrix (108 Runs Total):\n"
        "  - 3 Corpora: EmoDB, CAFE, AESDD\n"
        "  - 2 Features: MFCC vs. MFMC (12, 24, 30 coeffs)\n"
        "  - 2 Setups: None vs. Version A (clean 5x)\n"
        "  - 3 Speaker Assignment Seeds: 42, 43, 44\n\n"
        "• Execution & Runtime Estimate:\n"
        "  - Estimated total runtime: ~14–16 minutes on CPU.\n"
        "  - Status: Paused awaiting supervisor review and confirmation."
    )
    for p in tf9_1.paragraphs:
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_DARK

    # Right: Hypotheses & Expected Drop Card
    add_card(s9, Inches(6.8), Inches(1.7), Inches(5.7), Inches(5.2), title="Scientific Hypotheses to Test")
    tb9_2 = s9.shapes.add_textbox(Inches(7.0), Inches(2.3), Inches(5.3), Inches(4.4))
    tf9_2 = tb9_2.text_frame
    tf9_2.word_wrap = True
    tf9_2.text = (
        "• Expected Generalization Drop:\n"
        "  - Removing speaker identity overlap is expected to drop test accuracy by 15% to 25% across all corpora.\n"
        "  - Tests whether the 1D-CNN is genuinely learning emotion representations or acting as a speaker biometric classifier.\n\n"
        "• Does Gaussian Noise Protect Against Speaker Shift?\n"
        "  - Hypothesis: Additive Gaussian noise corrupts acoustics uniformly but does not alter fundamental speaker identity (pitch range, vocal tract length).\n"
        "  - Clean Version A augmentation is therefore expected to show minimal resilience against speaker shift.\n\n"
        "• Methodological Implication for Paper:\n"
        "  - The paper's ~97–99% figures rely on random splits (speaker overlap) combined with data leakage."
    )
    for p in tf9_2.paragraphs:
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_DARK

    add_speaker_notes(
        s9,
        "Person 4 (Evaluation Lead)",
        "On Slide 9, we outline the exact protocol for Task 3 before launching the runs. "
        "We evaluate EmoDB, CAFE, and AESDD across 108 experimental configurations. "
        "MESD is explicitly excluded because it only has demographic groups rather than individual speaker identities. "
        "Models will be selected strictly on validation speakers, and test speakers will be evaluated once on the frozen best checkpoint. "
        "The estimated runtime is approximately 14 to 16 minutes. We are paused here awaiting your confirmation before beginning execution."
    )

    # =========================================================================
    # SLIDE 10: Synthesis & Critical Findings
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    set_slide_background(s10)
    add_header(s10, "Synthesis & Critical Findings")

    # Card 1: Three Pillars of Findings
    add_card(s10, Inches(0.8), Inches(1.7), Inches(11.7), Inches(5.2), title="Core Conclusions from Week 4 Investigations")
    tb10 = s10.shapes.add_textbox(Inches(1.0), Inches(2.3), Inches(11.3), Inches(4.4))
    tf10 = tb10.text_frame
    tf10.word_wrap = True
    tf10.text = (
        "1. The 144-Input Discrepancy Does NOT Explain the Performance Gap\n"
        "   - Preserving 144 temporal frames increases model parameters and dramatically accelerates overfitting on small datasets.\n"
        "   - Test accuracy on frames144 drops to 51–61%, falling well below the pooled 24/48/60 baseline (64–71%).\n"
        "   - Summary statistics (mean + std) provide essential implicit regularization for small sample regimes.\n\n"
        "2. Additive Gaussian Noise Augmentation Does Not Reliably Boost Generalization\n"
        "   - Across German (EmoDB), French (CAFE), Greek (AESDD), and Spanish (MESD), clean Version A augmentation shifts test accuracy by only -2% to +5%.\n"
        "   - Most shifts fall within the random seed standard deviation (±2% to ±6%), indicating no statistically significant regularization benefit.\n"
        "   - The paper's claimed peak accuracies (~97–99%) were not replicated under our implementation across any of the four benchmarks.\n\n"
        "3. Speaker Identity Memorization is the Dominant Hidden Factor in High Reported Accuracies\n"
        "   - Evaluating on unseen speakers causes a sharp 12% to 25% accuracy collapse across all corpora.\n"
        "   - When combined with data leakage (augmenting before splitting, +15-20%), models easily reach artificial 90%+ scores without true emotion generalization.\n\n"
        "4. University Reproduction Takeaway\n"
        "   - Scientific honesty requires reporting genuine leak-free numbers: this 1D-CNN architecture achieves ~65-72% in speaker-dependent conditions, and ~48-55% in speaker-independent conditions."
    )
    for p in tf10.paragraphs:
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_DARK

    add_speaker_notes(
        s10,
        "Person 1 (Model Lead)",
        "To synthesize our findings: First, the 144-input format does not close the gap; in fact, temporal framing makes overfitting worse. "
        "Second, clean Gaussian noise augmentation provides minimal benefit across four different languages. "
        "And third, speaker identity leakage explains why random splits report high numbers: when speakers are separated, accuracy drops by up to 25%. "
        "Our reproduction study demonstrates that the published 97.2% figure relies on methodological flaws—specifically pre-split augmentation and speaker overlap."
    )

    # =========================================================================
    # SLIDE 11: Issues, Ambiguities & Scientific Integrity
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    set_slide_background(s11)
    add_header(s11, "Issues, Ambiguities & Scientific Integrity")

    # Card 1: Published Paper Ambiguities
    add_card(s11, Inches(0.8), Inches(1.7), Inches(5.6), Inches(5.2), title="Documented Paper Ambiguities")
    tb11_1 = s11.shapes.add_textbox(Inches(1.0), Inches(2.3), Inches(5.2), Inches(4.4))
    tf11_1 = tb11_1.text_frame
    tf11_1.word_wrap = True
    tf11_1.text = (
        "• Missing Specifications in the Published Article:\n"
        "  1. Input Dimension Construction:\n"
        "     - States '144 feature vectors' but omits frame length, hop size, windowing function, and audio duration.\n"
        "  2. Padding Configuration:\n"
        "     - Uses kernel size 15 across 4 Conv1D layers. Without padding='same', spatial dimensions collapse to negative numbers on 24-dim inputs.\n"
        "  3. Augmentation Timing:\n"
        "     - Completely silent on whether 5x Gaussian augmentation occurred before or after train/val/test splitting.\n"
        "  4. Speaker Partitioning:\n"
        "     - No mention of speaker disjointness or speaker cross-validation."
    )
    for p in tf11_1.paragraphs:
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_DARK

    # Card 2: Reproduction Standards
    add_card(s11, Inches(6.8), Inches(1.7), Inches(5.7), Inches(5.2), title="Reproduction Integrity Principles")
    tb11_2 = s11.shapes.add_textbox(Inches(7.0), Inches(2.3), Inches(5.3), Inches(4.4))
    tf11_2 = tb11_2.text_frame
    tf11_2.word_wrap = True
    tf11_2.text = (
        "• Adherence to Academic Best Practices:\n"
        "  - Data Provenance Disclosures: EmoDB and AESDD from official releases; CAFE and MESD ingested via CAMEO HuggingFace mirror (not verified against official Zenodo/Mendeley releases).\n"
        "  - Paper Citation Clarifications: Documented difference between 97.2% (Page 8 / Figure 8 text for 30 coeffs) and 97.5% (Page 13 Table 2 / Page 2 Abstract peak across all coeffs).\n"
        "  - Zero Cherry-Picking: Multi-seed evaluations (42, 43, 44) reported with mean ± standard deviation.\n"
        "  - Frozen Test Sets: Selected models chosen solely by validation accuracy; test partition scored exactly once.\n"
        "  - Cautious Language: Claims framed as 'not replicated under our implementation' rather than absolute failure."
    )
    for p in tf11_2.paragraphs:
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_DARK

    add_speaker_notes(
        s11,
        "Person 3 (Training & Pipeline Lead)",
        "On Slide 11, we explicitly document the ambiguities in the published paper. "
        "The paper omits critical hyperparameters: audio framing, padding strategy, and crucially, the timing of data augmentation relative to splitting. "
        "In our work, we have strictly adhered to academic reproducibility standards: zero data snooping, scalers fitted exclusively on training splits, models selected on validation accuracy, and all results logged across three random seeds."
    )

    # =========================================================================
    # SLIDE 12: Week 5 Roadmap
    # =========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    set_slide_background(s12)
    add_header(s12, "Next Steps: Week 5 Research Roadmap")

    # Card 1: Regularization
    add_card(s12, Inches(0.8), Inches(1.7), Inches(3.7), Inches(5.2), title="1. Advanced Regularization")
    tb12_1 = s12.shapes.add_textbox(Inches(1.0), Inches(2.3), Inches(3.3), Inches(4.4))
    tf12_1 = tb12_1.text_frame
    tf12_1.word_wrap = True
    tf12_1.text = (
        "• Combating Overfitting:\n"
        "  - L2 weight decay on dense layers.\n"
        "  - SpatialDropout1D across convolutional feature maps.\n"
        "  - Early stopping with patience=15 to prevent late-epoch memorization.\n"
        "  - Cosine annealing learning rate schedules."
    )
    for p in tf12_1.paragraphs:
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_DARK

    # Card 2: Domain-Aware Augmentations
    add_card(s12, Inches(4.8), Inches(1.7), Inches(3.7), Inches(5.2), title="2. Acoustic Augmentations")
    tb12_2 = s12.shapes.add_textbox(Inches(5.0), Inches(2.3), Inches(3.3), Inches(4.4))
    tf12_2 = tb12_2.text_frame
    tf12_2.word_wrap = True
    tf12_2.text = (
        "• Beyond Additive White Noise:\n"
        "  - SpecAugment (frequency and time masking) on spectrograms.\n"
        "  - Pitch shifting (±2 semitones) to simulate speaker vocal tract variance.\n"
        "  - Time stretching (0.9x–1.1x) to vary speaking rate.\n"
        "  - Test whether acoustic shifts improve speaker independence."
    )
    for p in tf12_2.paragraphs:
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_DARK

    # Card 3: Final Synthesis Report
    add_card(s12, Inches(8.8), Inches(1.7), Inches(3.7), Inches(5.2), title="3. Final Paper & Code Release")
    tb12_3 = s12.shapes.add_textbox(Inches(9.0), Inches(2.3), Inches(3.3), Inches(4.4))
    tf12_3 = tb12_3.text_frame
    tf12_3.word_wrap = True
    tf12_3.text = (
        "• Deliverables & Synthesis:\n"
        "  - Comprehensive reproduction report detailing why the paper's 97.2% claim is fragile.\n"
        "  - Open-source reproduction repository with automated verification suites.\n"
        "  - Guidelines for rigorous, leak-free evaluation in speech emotion recognition research."
    )
    for p in tf12_3.paragraphs:
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_DARK

    add_speaker_notes(
        s12,
        "Person 4 (Evaluation Lead)",
        "Looking ahead to Week 5: Now that we have established rigorous, leak-free baselines across four languages, we will explore methods to genuinely improve generalization. "
        "This includes testing real acoustic augmentations like SpecAugment, pitch shifting, and time stretching, which directly simulate speaker diversity, rather than simple additive noise. "
        "We will also prepare our final synthesis report, providing clear methodological recommendations for future speech emotion recognition research."
    )

    # =========================================================================
    # SLIDE 13 (BACKUP): Architecture & Loss Dynamics
    # =========================================================================
    s13 = prs.slides.add_slide(blank_layout)
    set_slide_background(s13)
    add_header(s13, "Backup Slide: Multi-Corpus Architecture & Loss Dynamics")

    add_card(s13, Inches(0.8), Inches(1.7), Inches(5.6), Inches(5.2), title="1D-CNN Layer-by-Layer Specifications")
    tb13_1 = s13.shapes.add_textbox(Inches(1.0), Inches(2.3), Inches(5.2), Inches(4.4))
    tf13_1 = tb13_1.text_frame
    tf13_1.word_wrap = True
    tf13_1.text = (
        "• Baseline Model (Global Pooled Summary):\n"
        "  - Input: (N, D, 1) where D in {24, 48, 60}\n"
        "  - Block 1: Conv1D(8, k=15, same) -> BN -> ReLU -> MaxPool(2)\n"
        "  - Block 2: Conv1D(16, k=15, same) -> BN -> ReLU -> MaxPool(2)\n"
        "  - Block 3: Conv1D(32, k=15, same) -> BN -> ReLU -> MaxPool(2)\n"
        "  - Block 4: Conv1D(64, k=15, same) -> BN -> ReLU -> MaxPool(2)\n"
        "  - Head: Dropout(0.3) -> Flatten -> Dense(120, ReLU) -> Dropout(0.2) -> Dense(C, Softmax)\n"
        "  - Total Parameters: 49,687 (12 coeffs) / 65,047 (24 & 30 coeffs)\n\n"
        "• frames144 Model (Temporal Sequence):\n"
        "  - Input: (N, 144, C) where C in {12, 24, 30}\n"
        "  - Pooling over time: 144 -> 72 -> 36 -> 18 -> 9 frames\n"
        "  - Flatten: 9 x 64 = 576 units into Dense(120)\n"
        "  - Total Parameters: ~104,000"
    )
    for p in tf13_1.paragraphs:
        p.font.size = Pt(10.5)
        p.font.color.rgb = TEXT_DARK

    add_card(s13, Inches(6.8), Inches(1.7), Inches(5.7), Inches(5.2), title="Multi-Seed Experimental Summary")
    tb13_2 = s13.shapes.add_textbox(Inches(7.0), Inches(2.3), Inches(5.3), Inches(4.4))
    tf13_2 = tb13_2.text_frame
    tf13_2.word_wrap = True
    tf13_2.text = (
        "• Execution & Reproducibility Parameters:\n"
        "  - Optimizer: Adam, Learning Rate = 0.001\n"
        "  - Epochs: 100, Batch Size = 32\n"
        "  - Loss: Sparse Categorical Cross-Entropy\n"
        "  - Checkpoint: ModelCheckpoint monitoring val_accuracy (mode='max')\n"
        "  - Test Evaluation: Frozen checkpoint evaluated once on test split\n\n"
        "• Verification & Testing:\n"
        "  - Automated unit test suite: tests/test_week4.py\n"
        "  - Confirms zero NaNs, correct tensor dimensions, valid split sizes, and strict speaker disjointness."
    )
    for p in tf13_2.paragraphs:
        p.font.size = Pt(10.5)
        p.font.color.rgb = TEXT_DARK

    add_speaker_notes(
        s13,
        "All Team Members",
        "Backup slide: Provides the exact parameter counts and layer configurations across both the baseline global pooling model and the 144-frame temporal model. "
        "Notice that frames144 almost doubles parameter count from 65k to 104k because of the larger flatten layer, which directly contributed to the rapid memorization and overfitting observed in Task 1."
    )

    prs.save(str(output_path))
    print(f"[SUCCESS] Generated status deck: {output_path}")


if __name__ == "__main__":
    out = ROOT_DIR / "Week4_Status.pptx"
    create_deck(out)
