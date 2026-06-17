"""
pdf_report.py — Generates a styled CancerAI diagnostic PDF report.
Uses only reportlab (pip install reportlab).
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, HRFlowable)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfgen import canvas
from datetime import datetime
import os

# ── Colour palette ────────────────────────────────────────────────────────
NAVY    = colors.HexColor('#0d1220')
BLUE    = colors.HexColor('#3b82f6')
INDIGO  = colors.HexColor('#6366f1')
RED     = colors.HexColor('#f87171')
GREEN   = colors.HexColor('#34d399')
GREY1   = colors.HexColor('#1a1e25')
GREY2   = colors.HexColor('#7a8aaa')
WHITE   = colors.white
LIGHT   = colors.HexColor('#e2e8f8')
RED_BG  = colors.HexColor('#2d1414')
GRN_BG  = colors.HexColor('#0d2318')

OUTPUT  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report.pdf")


def _page_bg(canv, doc):
    """Draw dark background + header bar on every page."""
    W, H = A4
    canv.saveState()

    # Dark background
    canv.setFillColor(NAVY)
    canv.rect(0, 0, W, H, fill=1, stroke=0)

    # Top accent bar
    canv.setFillColor(BLUE)
    canv.rect(0, H - 14*mm, W, 14*mm, fill=1, stroke=0)

    # Gradient-feel strip
    canv.setFillColor(INDIGO)
    canv.rect(W//2, H - 14*mm, W, 14*mm, fill=1, stroke=0)

    # Header text
    canv.setFillColor(WHITE)
    canv.setFont("Helvetica-Bold", 11)
    canv.drawString(20*mm, H - 9*mm, "CancerAI Diagnostics")
    canv.setFont("Helvetica", 9)
    canv.drawRightString(W - 20*mm, H - 9*mm, "Confidential — Research Use Only")

    # Footer
    canv.setFillColor(GREY2)
    canv.setFont("Helvetica", 8)
    canv.drawString(20*mm, 8*mm,
                    f"Generated: {datetime.now().strftime('%B %d, %Y  %H:%M')}")
    canv.drawRightString(W - 20*mm, 8*mm, f"Page {doc.page}")

    canv.restoreState()


def create_report(label: str, confidence: float, status: str,
                  organ: str = "",
                  class_id: str = "",
                  model_used: str = "",
                  probabilities: dict = None,
                  patient_id: str = "N/A",
                  output_path: str = None):
    """
    Generate a PDF diagnostic report.

    Parameters
    ----------
    label       : Human-readable class label  e.g. "Lung Adenocarcinoma"
    confidence  : Confidence percentage        e.g. 88.7
    status      : "CANCER" or "NORMAL"
    organ       : "Lung" / "Colon"
    class_id    : Internal class ID            e.g. "lung_aca"
    model_used  : Model description string
    probabilities : {class_name: probability_float} dict (optional)
    patient_id  : Patient / sample identifier (optional)
    output_path : Override default output path (optional)
    """

    out = output_path or OUTPUT
    W, H = A4

    doc = SimpleDocTemplate(
        out,
        pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=22*mm, bottomMargin=18*mm,
    )

    # ── Styles (FIXED - no duplicate fontName) ──────────────────────────────
    title_style = ParagraphStyle(
        "title",
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=28,
        textColor=WHITE,
        spaceAfter=4
    )

    sub_style = ParagraphStyle(
        "sub",
        fontName="Helvetica",
        fontSize=11,
        textColor=GREY2,
        spaceAfter=20
    )

    section_style = ParagraphStyle(
        "section",
        fontName="Helvetica-Bold",
        fontSize=10,
        textColor=BLUE,
        spaceBefore=14,
        spaceAfter=6,
        letterSpacing=1.5
    )

    body_style = ParagraphStyle(
        "body",
        fontName="Helvetica",
        fontSize=10,
        leading=16
    )

    label_style = ParagraphStyle(
        "lbl",
        fontName="Helvetica",
        fontSize=9,
        textColor=GREY2
    )

    value_style = ParagraphStyle(
        "val",
        fontName="Helvetica-Bold",
        fontSize=11,
        textColor=WHITE
    )

    disclaimer_sty = ParagraphStyle(
        "disc",
        fontName="Helvetica",
        fontSize=8,
        textColor=GREY2,
        leading=12,
        spaceBefore=16
    )
    
    # Style for verdict paragraph
    verdict_style = ParagraphStyle(
        "v",
        fontName="Helvetica-Bold",
        textColor=RED if status.upper() == "CANCER" else GREEN,
        leading=18
    )
    
    confidence_style = ParagraphStyle(
        "c",
        fontName="Helvetica-Bold",
        textColor=RED if status.upper() == "CANCER" else GREEN,
        alignment=TA_RIGHT,
        leading=32
    )

    story = []
    story.append(Spacer(1, 8*mm))

    # ── Title ─────────────────────────────────────────────────────────────
    story.append(Paragraph("Diagnostic Report", title_style))
    story.append(Paragraph("AI-Powered Cancer Histopathology Analysis", sub_style))
    story.append(HRFlowable(width="100%", thickness=1,
                             color=BLUE, spaceAfter=16))

    # ── Verdict box ───────────────────────────────────────────────────────
    is_cancer    = status.upper() == "CANCER"
    verdict_bg   = RED_BG if is_cancer else GRN_BG
    verdict_col  = RED if is_cancer else GREEN
    verdict_icon = "⚠  CANCER DETECTED" if is_cancer else "✓  NO CANCER DETECTED"

    verdict_data = [[
        Paragraph(f'<font size="14"><b>{verdict_icon}</b></font>', verdict_style),
        Paragraph(f'<font size="28"><b>{confidence:.1f}%</b></font>', confidence_style),
    ]]

    verdict_table = Table(verdict_data,
                          colWidths=[(W - 40*mm) * 0.65,
                                     (W - 40*mm) * 0.35])
    verdict_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), verdict_bg),
        ("ROUNDEDCORNERS", [8]),
        ("LEFTPADDING",  (0,0), (-1,-1), 14),
        ("RIGHTPADDING", (0,0), (-1,-1), 14),
        ("TOPPADDING",   (0,0), (-1,-1), 12),
        ("BOTTOMPADDING",(0,0), (-1,-1), 12),
        ("ALIGN", (1,0), (1,0), "RIGHT"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LINEBELOW", (0,0), (-1,-1), 1, verdict_col),
    ]))
    story.append(verdict_table)
    story.append(Spacer(1, 10*mm))

    # ── Classification details ────────────────────────────────────────────
    story.append(Paragraph("CLASSIFICATION DETAILS", section_style))

    fields = [
        ("Cancer Type",   label or "—"),
        ("Organ",         organ or "—"),
        ("Class ID",      class_id or "—"),
        ("Status",        status or "—"),
        ("Model",         model_used or "EfficientNet-B3 + miRNA Ensemble"),
        ("Patient ID",    patient_id),
        ("Analysis Date", datetime.now().strftime("%B %d, %Y")),
        ("Analysis Time", datetime.now().strftime("%H:%M:%S")),
    ]

    detail_rows = []
    for lbl, val in fields:
        detail_rows.append([
            Paragraph(lbl, label_style),
            Paragraph(str(val), value_style),
        ])

    detail_table = Table(
        detail_rows,
        colWidths=[(W - 40*mm) * 0.38, (W - 40*mm) * 0.62]
    )
    detail_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), GREY1),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [GREY1, colors.HexColor('#141926')]),
        ("LEFTPADDING",  (0,0), (-1,-1), 12),
        ("RIGHTPADDING", (0,0), (-1,-1), 12),
        ("TOPPADDING",   (0,0), (-1,-1), 8),
        ("BOTTOMPADDING",(0,0), (-1,-1), 8),
        ("LINEBELOW", (0,0), (-1,-1), 0.5, colors.HexColor('#1f2535')),
    ]))
    story.append(detail_table)
    story.append(Spacer(1, 8*mm))

    # ── Probability bars ──────────────────────────────────────────────────
    if probabilities:
        story.append(Paragraph("CLASS PROBABILITIES", section_style))

        bar_rows = []
        sorted_probs = sorted(probabilities.items(),
                              key=lambda x: x[1], reverse=True)
        max_w = W - 40*mm - 80*mm   # bar column width

        for cls, prob in sorted_probs:
            pct      = round(prob * 100, 1)
            bar_pct  = max(2, pct)          # min visible bar
            bar_col  = RED if pct > 50 and is_cancer else GREEN if pct > 50 else GREY2

            # Draw bar as a nested single-cell table
            bar_inner = Table([[""]], colWidths=[max_w * bar_pct / 100])
            bar_inner.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,-1), bar_col),
                ("TOPPADDING",    (0,0),(-1,-1), 4),
                ("BOTTOMPADDING", (0,0),(-1,-1), 4),
            ]))

            bar_rows.append([
                Paragraph(cls.replace("_", " ").title(),
                          ParagraphStyle("bn", fontName="Helvetica",
                                         textColor=LIGHT, fontSize=9)),
                bar_inner,
                Paragraph(f"{pct}%",
                          ParagraphStyle("bp", fontName="Helvetica-Bold",
                                         textColor=WHITE, fontSize=9,
                                         alignment=TA_RIGHT)),
            ])

        bar_table = Table(
            bar_rows,
            colWidths=[52*mm, max_w, 18*mm]
        )
        bar_table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), GREY1),
            ("ROWBACKGROUNDS", (0,0), (-1,-1),
             [GREY1, colors.HexColor('#141926')]),
            ("LEFTPADDING",  (0,0), (-1,-1), 10),
            ("RIGHTPADDING", (0,0), (-1,-1), 10),
            ("TOPPADDING",   (0,0), (-1,-1), 6),
            ("BOTTOMPADDING",(0,0), (-1,-1), 6),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("LINEBELOW", (0,0), (-1,-1), 0.5,
             colors.HexColor('#1f2535')),
        ]))
        story.append(bar_table)
        story.append(Spacer(1, 8*mm))

    # ── Interpretation ────────────────────────────────────────────────────
    story.append(Paragraph("INTERPRETATION", section_style))

    tier = ("Very High" if confidence >= 90 else
            "High"      if confidence >= 75 else
            "Moderate"  if confidence >= 55 else "Low")

    if is_cancer:
        interp = (
            f"The AI model identified <b>{label}</b> with "
            f"<b>{confidence:.1f}% confidence</b> ({tier} confidence tier). "
            f"The tissue sample exhibits characteristics consistent with malignant pathology "
            f"in the <b>{organ or 'affected'}</b> organ. "
            f"This result should be reviewed by a qualified pathologist before any "
            f"clinical decision is made."
        )
    else:
        interp = (
            f"The AI model classified this sample as <b>{label}</b> with "
            f"<b>{confidence:.1f}% confidence</b> ({tier} confidence tier). "
            f"No malignant characteristics were detected in this analysis. "
            f"Regular screening is still recommended as per clinical guidelines."
        )

    story.append(Paragraph(interp,
                            ParagraphStyle("interp", fontName="Helvetica",
                                           textColor=LIGHT, fontSize=10,
                                           leading=16,
                                           backColor=GREY1,
                                           leftIndent=12, rightIndent=12,
                                           spaceAfter=0,
                                           borderPad=12)))
    story.append(Spacer(1, 8*mm))

    # ── Disclaimer ────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=GREY2, spaceAfter=8))
    story.append(Paragraph(
        "⚠  DISCLAIMER: This report is generated by an AI system for research and "
        "educational purposes only. It does not constitute a medical diagnosis. "
        "Results must be interpreted by a qualified medical professional. "
        "CancerAI Diagnostics accepts no liability for clinical decisions made "
        "based on this report.",
        disclaimer_sty
    ))

    doc.build(story, onFirstPage=_page_bg, onLaterPages=_page_bg)
    return out


# ── CLI test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    path = create_report(
        label="Lung Adenocarcinoma",
        confidence=88.7,
        status="CANCER",
        organ="Lung",
        class_id="lung_aca",
        model_used="EfficientNet-B3 (8× TTA)",
        probabilities={
            "lung_aca": 0.887,
            "lung_n":   0.042,
            "lung_scc": 0.031,
            "colon_aca":0.024,
            "colon_n":  0.016,
        },
        patient_id="SAMPLE-001",
    )
    print(f"Report saved → {path}")