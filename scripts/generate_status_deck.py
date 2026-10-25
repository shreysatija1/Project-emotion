"""
Generate Week 3 Status Presentation (Week3_Status.pptx) using python-pptx.
Adheres strictly to the guidelines:
  - 16:9 widescreen layout
  - Clean, modern, accessible design (clear contrast, card layouts)
  - Real figures from existing results PNGs
  - Plain language for non-technical audience
  - Softened claims matching Task 2
  - Speaker notes split across 4 presenters (Model, Augmentation, Training, Evaluation)
  - Backup slide with layer details and curves
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

# Color Palette
BG_COLOR = RGBColor(248, 250, 252)        # Slate 50
CARD_BG = RGBColor(255, 255, 255)         # White
PRIMARY = RGBColor(30, 58, 138)           # Blue 900
SECONDARY = RGBColor(14, 116, 144)        # Cyan 700
ACCENT = RGBColor(37, 99, 235)            # Blue 600
TEXT_DARK = RGBColor(15, 23, 42)          # Slate 900
TEXT_MUTED = RGBColor(71, 85, 105)        # Slate 600
BORDER_COLOR = RGBColor(226, 232, 240)    # Slate 200
HIGHLIGHT_BG = RGBColor(238, 242, 255)    # Indigo 50


def set_slide_background(slide, color=BG_COLOR):
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_header(slide, title_text, category_text="SPEECH EMOTION RECOGNITION • WEEK 3"):
    # Category Tracker
    cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.5), Inches(0.4))
    tf_c = cat_box.text_frame
    tf_c.word_wrap = True
    p_c = tf_c.paragraphs[0]
    p_c.text = category_text.upper()
    p_c.font.size = Pt(10)
    p_c.font.bold = True
    p_c.font.color.rgb = SECONDARY

    # Title
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


def create_deck(output_path="Week3_Status.pptx"):
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # =========================================================================
    # SLIDE 1: Title Slide (Presenter 1: Model)
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_background(s1, RGBColor(241, 245, 249))

    # Center card
    add_card(s1, Inches(1.5), Inches(1.2), Inches(10.333), Inches(5.1), bg_color=CARD_BG, border_color=BORDER_COLOR)

    tb = s1.shapes.add_textbox(Inches(2.0), Inches(1.8), Inches(9.333), Inches(3.8))
    tf = tb.text_frame
    tf.word_wrap = True

    p0 = tf.paragraphs[0]
    p0.text = "RESEARCH PAPER REPRODUCTION • WEEK 3"
    p0.font.size = Pt(12)
    p0.font.bold = True
    p0.font.color.rgb = SECONDARY
    p0.space_after = Pt(14)

    p1 = tf.add_paragraph()
    p1.text = "Speech Emotion Recognition with 1D-CNN"
    p1.font.size = Pt(32)
    p1.font.bold = True
    p1.font.color.rgb = PRIMARY
    p1.space_after = Pt(10)

    p2 = tf.add_paragraph()
    p2.text = "Evaluating Audio Data Augmentation and Feature Representations on EmoDB"
    p2.font.size = Pt(18)
    p2.font.color.rgb = TEXT_MUTED
    p2.space_after = Pt(28)

    p3 = tf.add_paragraph()
    p3.text = "Reproduction of: Thomas Mary Little Flower et al. (Automatika 2024)\nTeam Emotion • Project Status & Findings"
    p3.font.size = Pt(14)
    p3.font.color.rgb = TEXT_DARK

    add_speaker_notes(
        s1,
        "Person 1 (Model)",
        "Welcome everyone. Today we are presenting our Week 3 status on reproducing the 2024 Automatika paper on Speech Emotion Recognition. "
        "In this phase, we implemented the paper's deep 1D Convolutional Neural Network architecture, established strict anti-leakage training protocols, "
        "and carried out 54 controlled experiments to test whether noise augmentation and new feature types truly boost emotion detection."
    )

    # =========================================================================
    # SLIDE 2: Recap: Weeks 1–2 & Week 3 Goals (Presenter 1: Model)
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    set_slide_background(s2)
    add_header(s2, "Project Roadmap: From Data Pipeline to Deep Learning")

    # Card 1: Week 1
    add_card(s2, Inches(0.8), Inches(1.8), Inches(3.6), Inches(4.8), title="Week 1: Foundations")
    tb1 = s2.shapes.add_textbox(Inches(1.0), Inches(2.5), Inches(3.2), Inches(3.8))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    tf1.text = (
        "• Loaded 535 EmoDB audio clips across 7 emotions.\n\n"
        "• Implemented MFCC and MFMC feature extraction (12, 24, 30 coefficients).\n\n"
        "• Established stratified 60/20/20 split.\n\n"
        "• Classical baselines: SVM reached 88.8% test accuracy without augmentation."
    )
    for p in tf1.paragraphs:
        p.font.size = Pt(13)
        p.font.color.rgb = TEXT_DARK

    # Card 2: Week 2
    add_card(s2, Inches(4.85), Inches(1.8), Inches(3.6), Inches(4.8), title="Week 2: Augmentation")
    tb2 = s2.shapes.add_textbox(Inches(5.05), Inches(2.5), Inches(3.2), Inches(3.8))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    tf2.text = (
        "• Built 5x Gaussian noise augmentation pipeline (noise levels 0.01, 0.025, 0.05).\n\n"
        "• Created Version A: augment training clips only (clean protocol, 1,819 rows).\n\n"
        "• Created Version B: augment all clips then split (potential leakage, 2,675 rows).\n\n"
        "• Verified zero feature NaNs and bitwise identical clean copies."
    )
    for p in tf2.paragraphs:
        p.font.size = Pt(13)
        p.font.color.rgb = TEXT_DARK

    # Card 3: Week 3
    add_card(s2, Inches(8.9), Inches(1.8), Inches(3.6), Inches(4.8), title="Week 3: Deep Learning Goal", bg_color=HIGHLIGHT_BG)
    tb3 = s2.shapes.add_textbox(Inches(9.1), Inches(2.5), Inches(3.2), Inches(3.8))
    tf3 = tb3.text_frame
    tf3.word_wrap = True
    tf3.text = (
        "• Build and compile the paper's 1D-CNN architecture.\n\n"
        "• Execute 18 experimental configurations across 3 independent random seeds (54 runs).\n\n"
        "• Strictly prevent evaluation data contamination.\n\n"
        "• Scientifically evaluate the paper's high accuracy claims (86.9% baseline, 97.2% with augmentation)."
    )
    for p in tf3.paragraphs:
        p.font.size = Pt(13)
        p.font.color.rgb = PRIMARY

    add_speaker_notes(
        s2,
        "Person 1 (Model)",
        "To orient everyone on our progress: In Week 1, we processed the raw EmoDB audio, extracted MFCC and MFMC features, and verified with basic SVMs that the signal contains emotional information. "
        "In Week 2, we created the 5-set noise augmentation datasets and set up two versions: Version A, which only augments the training set, and Version B, which augments before splitting. "
        "Our mission in Week 3 was to train the 1D-CNN across all setups and compare the results with the paper."
    )

    # =========================================================================
    # SLIDE 3: The Problem & Pipeline in One Picture (Presenter 1: Model)
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    set_slide_background(s3)
    add_header(s3, "The Speech Emotion Recognition Pipeline")

    # Step boxes
    steps = [
        ("1. Speech Audio", "535 raw German utterances\n(16 kHz Mono, 10 speakers)", Inches(0.8)),
        ("2. Acoustic Features", "MFCC (cepstral) &\nMFMC (magnitude)\n12, 24, 30 coefficients", Inches(3.85)),
        ("3. 1D-CNN Model", "4 Convolutional Stages\nPattern detection &\nTemporal pooling", Inches(6.9)),
        ("4. Emotion Class", "Predicts 1 of 7 emotions:\nAnger, Boredom, Disgust,\nFear, Happy, Neutral, Sad", Inches(9.95)),
    ]

    for title, desc, left in steps:
        add_card(s3, left, Inches(2.0), Inches(2.6), Inches(4.4), title=title)
        tb = s3.shapes.add_textbox(left + Inches(0.15), Inches(2.8), Inches(2.3), Inches(3.0))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.text = desc
        for p in tf.paragraphs:
            p.font.size = Pt(13)
            p.font.color.rgb = TEXT_DARK

    # Arrow symbols between boxes
    for arrow_left in [Inches(3.45), Inches(6.5), Inches(9.55)]:
        arr_box = s3.shapes.add_textbox(arrow_left, Inches(3.8), Inches(0.4), Inches(0.5))
        p = arr_box.text_frame.paragraphs[0]
        p.text = "➔"
        p.font.size = Pt(24)
        p.font.color.rgb = SECONDARY

    add_speaker_notes(
        s3,
        "Person 1 (Model)",
        "This slide illustrates the full end-to-end journey of an audio clip in plain terms. "
        "First, we take an emotional recording spoken by an actor. "
        "Second, we transform the complex raw sound wave into acoustic frequency representations: either standard MFCCs or magnitude-based MFMCs. "
        "Third, these features are fed into a 1D Convolutional Neural Network that automatically detects emotional patterns. "
        "Finally, the network outputs a probability score across seven emotions to make its final guess."
    )

    # =========================================================================
    # SLIDE 4: The 1D-CNN Model in Plain Words (Presenter 1: Model)
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    set_slide_background(s4)
    add_header(s4, "The 1D-CNN Model Explained Simply")

    # Left Card: How it works
    add_card(s4, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.8), title="What the Model Does")
    tb_l = s4.shapes.add_textbox(Inches(1.0), Inches(2.5), Inches(5.2), Inches(3.8))
    tf_l = tb_l.text_frame
    tf_l.word_wrap = True
    tf_l.text = (
        "• A Pattern Scanner: Just as image CNNs scan for edges and textures, a 1D-CNN scans across acoustic coefficients for characteristic emotional shapes.\n\n"
        "• 4 Successive Stages: The network starts with 8 pattern detectors, progressively widening to 16, 32, and 64 detectors to capture complex acoustic combinations.\n\n"
        "• Summary Pooling: After each scan, it compresses the features by half (MaxPooling) to keep only the strongest indicators.\n\n"
        "• Decision Layer: A dense layer of 120 neurons synthesizes these clues and outputs probabilities for the 7 emotion categories."
    )
    for p in tf_l.paragraphs:
        p.font.size = Pt(13)
        p.font.color.rgb = TEXT_DARK

    # Right Card: Plain Analogy & Architecture Highlights
    add_card(s4, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.8), title="Design Highlights & Analogy", bg_color=HIGHLIGHT_BG)
    tb_r = s4.shapes.add_textbox(Inches(7.0), Inches(2.5), Inches(5.3), Inches(3.8))
    tf_r = tb_r.text_frame
    tf_r.word_wrap = True
    tf_r.text = (
        "• The Analogy: Think of a sound engineer listening through four specialized equalizer filters. First they check broad pitch, then tonal harshness, then energy spread, and finally decide the speaker's emotional state.\n\n"
        "• Total Parameters: ~65,000 trainable weights—compact and fast enough to run on standard computers.\n\n"
        "• Regularization: Uses Dropout (randomly turning off neurons during training) and Batch Normalization to prevent the model from simply memorizing training clips."
    )
    for p in tf_r.paragraphs:
        p.font.size = Pt(13)
        p.font.color.rgb = PRIMARY

    add_speaker_notes(
        s4,
        "Person 1 (Model)",
        "For non-deep-learning audiences, the easiest way to understand a 1D Convolutional Neural Network is as a hierarchy of pattern scanners. "
        "Instead of scanning pixels on a photo, it scans across the acoustic frequencies of speech. "
        "It uses four stages, starting with simple acoustic shapes and building up to complex emotional signatures. "
        "It is relatively lightweight, with about 65,000 parameters, and uses Dropout and Batch Normalization to help it generalize."
    )

    # =========================================================================
    # SLIDE 5: Data Augmentation: Why and How (Presenter 2: Augmentation)
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    set_slide_background(s5)
    add_header(s5, "Data Augmentation: Teaching the Model Robustness")

    # Left text card
    add_card(s5, Inches(0.8), Inches(1.8), Inches(5.8), Inches(4.8), title="The 5-Set Noise Augmentation")
    tb5 = s5.shapes.add_textbox(Inches(1.0), Inches(2.5), Inches(5.4), Inches(3.8))
    tf5 = tb5.text_frame
    tf5.word_wrap = True
    tf5.text = (
        "• Why Augment? Deep learning models need large amounts of diverse data. EmoDB has only 535 clips, making models prone to memorization.\n\n"
        "• The 5 Augmented Copies per Recording:\n"
        "   1. Set 1: Original audio waveform (clean)\n"
        "   2. Set 2: Exact zero-noise copy of original\n"
        "   3. Set 3: Audio + 1% gentle background noise\n"
        "   4. Set 4: Audio + 2.5% moderate background noise\n"
        "   5. Set 5: Audio + 5.0% noticeable background noise\n\n"
        "• The Goal: Force the network to focus on underlying vocal emotion rather than delicate background silence or recording quirks."
    )
    for p in tf5.paragraphs:
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_DARK

    # Right side: Embed spectrogram / MFCC comparison image if available
    spec_img = RESULTS_DIR / "mfcc_vs_mfmc_examples.png"
    if spec_img.exists():
        add_card(s5, Inches(6.9), Inches(1.8), Inches(5.6), Inches(4.8), title="Acoustic Representations (MFCC vs MFMC)")
        s5.shapes.add_picture(str(spec_img), Inches(7.1), Inches(2.4), width=Inches(5.2))
    else:
        add_card(s5, Inches(6.9), Inches(1.8), Inches(5.6), Inches(4.8), title="5x Multiplied Data Volume")

    add_speaker_notes(
        s5,
        "Person 2 (Augmentation)",
        "Now turning to data augmentation. Neural networks easily memorize small datasets. With only 535 total audio files, a network can quickly overfit. "
        "To teach the network robustness, the paper specifies creating five versions of every audio clip: the original, a clean copy, and three versions with increasing levels of Gaussian white noise—1%, 2.5%, and 5%. "
        "The image on the right shows the acoustic fingerprints extracted from these audio clips."
    )

    # =========================================================================
    # SLIDE 6: Our Experiment & Anti-Leakage Protocol (Presenter 2: Augmentation)
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    set_slide_background(s6)
    add_header(s6, "Experimental Matrix & Preventing Data Leakage")

    # Grid summary card
    add_card(s6, Inches(0.8), Inches(1.8), Inches(3.6), Inches(4.8), title="The 54-Run Matrix")
    tb_g = s6.shapes.add_textbox(Inches(1.0), Inches(2.5), Inches(3.2), Inches(3.8))
    tf_g = tb_g.text_frame
    tf_g.word_wrap = True
    tf_g.text = (
        "• 2 Features: MFCC & MFMC\n\n"
        "• 3 Coeff Sizes: 12, 24, 30\n\n"
        "• 3 Setup Modes:\n"
        "   - None (original 321 train)\n"
        "   - Version A (clean after-split)\n"
        "   - Version B (augment-first)\n\n"
        "• 3 Random Seeds: 42, 43, 44\n\n"
        "• Total: 54 fully trained models (100 epochs each)."
    )
    for p in tf_g.paragraphs:
        p.font.size = Pt(13)
        p.font.color.rgb = TEXT_DARK

    # Version A card (Clean)
    add_card(s6, Inches(4.85), Inches(1.8), Inches(3.6), Inches(4.8), title="Version A: Honest Split", bg_color=HIGHLIGHT_BG)
    tb_a = s6.shapes.add_textbox(Inches(5.05), Inches(2.5), Inches(3.2), Inches(3.8))
    tf_a = tb_a.text_frame
    tf_a.word_wrap = True
    tf_a.text = (
        "• Split FIRST, then augment:\n"
        "   - Train (321 clips) -> 1,605\n"
        "   - Val (107 clips) -> clean only\n"
        "   - Test (107 clips) -> clean only\n\n"
        "• Strict Zero Leakage: 0% overlap between train and test recordings.\n\n"
        "• Standardized using training statistics only.\n\n"
        "• Measures true real-world generalization."
    )
    for p in tf_a.paragraphs:
        p.font.size = Pt(13)
        p.font.color.rgb = PRIMARY

    # Version B card (Leaked)
    add_card(s6, Inches(8.9), Inches(1.8), Inches(3.6), Inches(4.8), title="Version B: Augment-First")
    tb_b = s6.shapes.add_textbox(Inches(9.1), Inches(2.5), Inches(3.2), Inches(3.8))
    tf_b = tb_b.text_frame
    tf_b.word_wrap = True
    tf_b.text = (
        "• Augment FIRST, then split:\n"
        "   - 535 clips -> 2,675 rows\n"
        "   - Split 60/20/20 randomly\n\n"
        "• Pervasive Leakage:\n"
        "   - 95.89% of test clips share source audio with training!\n"
        "   - 96.64% of validation clips share source audio with training!\n\n"
        "• The network is tested on noisy copies of clips it already memorized."
    )
    for p in tf_b.paragraphs:
        p.font.size = Pt(13)
        p.font.color.rgb = TEXT_DARK

    add_speaker_notes(
        s6,
        "Person 2 (Augmentation)",
        "This slide highlights the central methodological distinction of our project. "
        "We ran 54 distinct training runs across two feature types, three coefficient sizes, and three random seeds. "
        "Most importantly, we tested two augmentation protocols: "
        "Version A splits the data before augmenting, so the test set has zero overlap with training. "
        "Version B augments all files first and then splits. In Version B, almost 96% of test clips are simply noisy replicas of training recordings. "
        "This contrast is crucial for interpreting published deep learning accuracy claims."
    )

    # =========================================================================
    # SLIDE 7: Results Table & Confusion Matrix (Presenter 3: Training)
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    set_slide_background(s7)
    add_header(s7, "Experimental Results & Confusion Matrix")

    # Embed table image on left
    tbl_img = RESULTS_DIR / "final_results_table.png"
    if tbl_img.exists():
        add_card(s7, Inches(0.8), Inches(1.8), Inches(6.8), Inches(4.8), title="Test & Validation Accuracy (Mean ± Std)")
        s7.shapes.add_picture(str(tbl_img), Inches(1.0), Inches(2.4), width=Inches(6.4))

    # Embed confusion matrix on right
    cm_img = RESULTS_DIR / "confusion_matrix_best_mfcc_after_split.png"
    if cm_img.exists():
        add_card(s7, Inches(7.9), Inches(1.8), Inches(4.6), Inches(4.8), title="Emotion Confusion Matrix (MFCC Best)")
        s7.shapes.add_picture(str(cm_img), Inches(8.1), Inches(2.4), width=Inches(4.2))

    add_speaker_notes(
        s7,
        "Person 3 (Training)",
        "Turning to training results: The table on the left presents the mean and standard deviation across all three seeds. "
        "Under our clean setup (Version A), MFCC test accuracy reaches about 70.7%, and MFMC reaches about 72.3%. "
        "However, when we look at Version B on the right column of the table, test accuracy jumps to nearly 89%. "
        "The confusion matrix on the right shows where the model succeeds and struggles: it recognizes Anger and Happiness with high precision, but frequently confuses Neutral, Boredom, and Sadness."
    )

    # =========================================================================
    # SLIDE 8: Findings (Presenter 3: Training)
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    set_slide_background(s8)
    add_header(s8, "Key Findings & Scientific Interpretations")

    add_card(s8, Inches(0.8), Inches(1.8), Inches(11.7), Inches(4.8), title="Observations and Evidence-Based Conclusions")
    tb8 = s8.shapes.add_textbox(Inches(1.1), Inches(2.4), Inches(11.1), Inches(4.0))
    tf8 = tb8.text_frame
    tf8.word_wrap = True
    tf8.text = (
        "1. Our leak-free result is far below the paper:\n"
        "   • Under clean after-split augmentation (Version A), our models reach 70.7% to 72.3% test accuracy, compared to the paper's reported 97.2% with augmentation.\n\n"
        "2. Leakage is a plausible contributing factor, but not the only difference:\n"
        "   • When we augment before splitting (Version B), test accuracy rises sharply to 87.2% – 89.3%. Because 95.9% of test files overlap with training, leakage is a plausible contributing factor.\n"
        "   • However, our setup also differs from the paper in other ways (input size 24/48/60 vs 144, 64 vs 62 filters, batch size, noise scaling). We could not find a clear statement in the paper of whether augmentation happened before or after the split.\n\n"
        "3. Leakage cannot explain the no-augmentation baseline gap:\n"
        "   • Our unaugmented baseline reaches 64% – 71% test accuracy, which is already far below the paper's 86.9% no-augmentation figure. Leakage cannot explain that gap since neither baseline contains augmented copies.\n\n"
        "4. Evidence suggests mixed augmentation effects:\n"
        "   • The test set has only 107 clips (one clip is ~0.9%) and standard deviations reach 4% – 7%. Only the MFMC 12-coefficient gain (+11.9%, 51.7% -> 63.6%) clearly stands out.\n"
        "   • The idea that DCT decorrelation makes MFCC less sensitive to noise regularization remains an untested hypothesis."
    )
    for p in tf8.paragraphs:
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_DARK

    add_speaker_notes(
        s8,
        "Person 3 (Training)",
        "Here are our key takeaways. First, our clean leak-free result is around 71 to 72%, which is well below the paper's claimed 97.2%. "
        "Second, when we augment before splitting, accuracy jumps sharply into the upper 80s, indicating that data leakage is a plausible contributing factor. "
        "However, our setup also differs in input dimensions and filter count, and we could not find an explicit statement in the paper confirming their split order. "
        "Third, leakage cannot explain why our baseline without augmentation is 64 to 71% compared to the paper's 86.9%. "
        "Finally, with 107 test clips and standard deviations of 4 to 7%, only the 12-coefficient MFMC gain stands out clearly."
    )

    # =========================================================================
    # SLIDE 9: Assumptions & Differences from the Paper (Presenter 4: Evaluation)
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    set_slide_background(s9)
    add_header(s9, "Assumptions & Engineering Decisions")

    items = [
        ("Input Dimension", "Paper states 144 features; we used 24, 48, and 60 (mean and std pooled over time for 12, 24, 30 coefficients)."),
        ("Convolution Padding", "Used 'same' padding. With 'valid' padding, sequence lengths collapse to zero by layer 3 given kernel size 15 and 4 pooling layers."),
        ("Layer 4 Filters", "Paper specifies 62 filters; we adopted 64 filters as the standard power-of-two following 8, 16, and 32."),
        ("Capacity Asymmetry", "12-coefficient model flattens to 64 units into Dense(120), while 24 and 30 flatten to 192 units. Comparisons are not purely about acoustic features."),
        ("Evaluation Protocol", "Best models selected strictly on validation accuracy; test set scored exactly once after freezing model selection."),
        ("Train-Only Scaling", "StandardScaler fitted strictly on training data; validation and test normalized without data peek."),
    ]

    for idx, (title, desc) in enumerate(items):
        row = idx // 2
        col = idx % 2
        left = Inches(0.8) if col == 0 else Inches(6.8)
        top = Inches(1.8 + row * 1.6)

        add_card(s9, left, top, Inches(5.7), Inches(1.4), title=title)
        tb = s9.shapes.add_textbox(left + Inches(0.15), top + Inches(0.55), Inches(5.4), Inches(0.75))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.text = desc
        for p in tf.paragraphs:
            p.font.size = Pt(11)
            p.font.color.rgb = TEXT_DARK

    add_speaker_notes(
        s9,
        "Person 4 (Evaluation)",
        "Turning to our methodology: To reproduce the paper faithfully, we had to resolve several technical ambiguities. "
        "The paper mentions 144 features, whereas 12, 24, and 30 coefficients yield 24, 48, and 60 pooled statistics. "
        "We also used 'same' padding so the 1D convolutions would not run out of dimensions, and corrected the 4th layer to 64 filters. "
        "Importantly, we note that the 12-coefficient model feeds into a smaller flatten layer than the 24 and 30 models, meaning capacity differences also play a role."
    )

    # =========================================================================
    # SLIDE 10: Next Steps: Week 4 Multi-Corpus Evaluation (Presenter 4: Evaluation)
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    set_slide_background(s10)
    add_header(s10, "Looking Ahead: Week 4 Multi-Corpus Evaluation")

    add_card(s10, Inches(0.8), Inches(1.8), Inches(3.6), Inches(4.8), title="CAFE Dataset", bg_color=HIGHLIGHT_BG)
    tb_c = s10.shapes.add_textbox(Inches(1.0), Inches(2.5), Inches(3.2), Inches(3.8))
    tf_c = tb_c.text_frame
    tf_c.word_wrap = True
    tf_c.text = (
        "• Canadian French Speech\n\n"
        "• 6 fundamental emotions + neutral\n\n"
        "• 12 professional actors\n\n"
        "• Goal: Test whether MFMC vs MFCC behavior holds in French."
    )
    for p in tf_c.paragraphs:
        p.font.size = Pt(13)
        p.font.color.rgb = PRIMARY

    add_card(s10, Inches(4.85), Inches(1.8), Inches(3.6), Inches(4.8), title="AESDD Dataset")
    tb_e = s10.shapes.add_textbox(Inches(5.05), Inches(2.5), Inches(3.2), Inches(3.8))
    tf_e = tb_e.text_frame
    tf_e.word_wrap = True
    tf_e.text = (
        "• Greek Emotional Speech\n\n"
        "• 5 distinct emotional states\n\n"
        "• Acted dramatic speech\n\n"
        "• Goal: Evaluate noise augmentation benefits on smaller phonetic corpora."
    )
    for p in tf_e.paragraphs:
        p.font.size = Pt(13)
        p.font.color.rgb = TEXT_DARK

    add_card(s10, Inches(8.9), Inches(1.8), Inches(3.6), Inches(4.8), title="MESD Dataset")
    tb_m = s10.shapes.add_textbox(Inches(9.1), Inches(2.5), Inches(3.2), Inches(3.8))
    tf_m = tb_m.text_frame
    tf_m.word_wrap = True
    tf_m.text = (
        "• Mexican Spanish Emotional Speech\n\n"
        "• Multiple speaker categories\n\n"
        "• Real-world acoustic variance\n\n"
        "• Goal: Investigate cross-corpus generalization and speaker independence."
    )
    for p in tf_m.paragraphs:
        p.font.size = Pt(13)
        p.font.color.rgb = TEXT_DARK

    add_speaker_notes(
        s10,
        "Person 4 (Evaluation)",
        "Finally, our plan for Week 4: EmoDB is a German-language dataset with 10 speakers. "
        "To test whether these conclusions hold globally, we will expand our verified pipeline to three additional corpora: "
        "CAFE in Canadian French, AESDD in Greek, and MESD in Mexican Spanish. "
        "This cross-lingual evaluation will help us determine if the augmentation patterns and feature dynamics we observed generalize across languages."
    )

    # =========================================================================
    # SLIDE 11 (BACKUP): Layer Architecture & Convergence Curves
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    set_slide_background(s11)
    add_header(s11, "Backup Slide: Layer Dimensions & Convergence Curves")

    # Left: Layer details
    add_card(s11, Inches(0.8), Inches(1.8), Inches(4.8), Inches(4.8), title="Layer-by-Layer Architecture")
    tb11 = s11.shapes.add_textbox(Inches(1.0), Inches(2.4), Inches(4.4), Inches(4.0))
    tf11 = tb11.text_frame
    tf11.word_wrap = True
    tf11.text = (
        "• Input: (batch, input_dim, 1) where input_dim in {24, 48, 60}\n"
        "• Block 1: Conv1D(8, k=15, same) -> BN -> ReLU -> MaxPool(2)\n"
        "• Block 2: Conv1D(16, k=15, same) -> BN -> ReLU -> MaxPool(2)\n"
        "• Block 3: Conv1D(32, k=15, same) -> BN -> ReLU -> MaxPool(2)\n"
        "• Block 4: Conv1D(64, k=15, same) -> BN -> ReLU -> MaxPool(2)\n"
        "• Head: Dropout(0.3) -> Flatten -> Dense(120, ReLU) -> Dropout(0.2) -> Dense(7, Softmax)\n"
        "• Total Parameters: 49,687 (12 coeffs) / 65,047 (24 & 30 coeffs)"
    )
    for p in tf11.paragraphs:
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_DARK

    # Right: Learning Curves
    curves_img = RESULTS_DIR / "training_curves_best_runs.png"
    if curves_img.exists():
        add_card(s11, Inches(5.9), Inches(1.8), Inches(6.6), Inches(4.8), title="Training & Validation Convergence")
        s11.shapes.add_picture(str(curves_img), Inches(6.1), Inches(2.4), width=Inches(6.2))

    add_speaker_notes(
        s11,
        "All Team Members",
        "Backup slide: Shows the exact layer parameters and training curves. Notice how training loss rapidly approaches zero across all runs, while validation loss diverges after epoch 20 to 30, illustrating that all models experience overfitting on this small dataset."
    )

    prs.save(str(output_path))
    print(f"[SUCCESS] Generated status deck: {output_path}")


if __name__ == "__main__":
    out = ROOT_DIR / "Week3_Status.pptx"
    create_deck(out)
