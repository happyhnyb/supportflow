"""Render the Markdown report to a polished, self-contained PDF."""

from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "report.md"
OUTPUT = ROOT / "output" / "pdf" / "supportflow_report.pdf"


def clean_inline(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", text)
    text = re.sub(
        r"(https?://[^\s<]+)",
        r"<link href='\1' color='#0F766E'>\1</link>",
        text,
    )
    return text


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#CBD5E1"))
    canvas.line(2 * cm, 1.55 * cm, A4[0] - 2 * cm, 1.55 * cm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#475569"))
    canvas.drawString(2 * cm, 1.05 * cm, "SupportFlow - AI application design report")
    canvas.drawRightString(A4[0] - 2 * cm, 1.05 * cm, f"Page {doc.page}")
    canvas.restoreState()


def make_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=24,
                              leading=29, textColor=colors.HexColor("#0F172A"), alignment=TA_CENTER, spaceAfter=14))
    styles.add(ParagraphStyle(name="Subtitle", parent=styles["Normal"], fontSize=11, leading=16,
                              textColor=colors.HexColor("#475569"), alignment=TA_CENTER, spaceAfter=8))
    styles.add(ParagraphStyle(name="H1Report", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=16,
                              leading=21, textColor=colors.HexColor("#0F766E"), spaceBefore=14, spaceAfter=7,
                              keepWithNext=True))
    styles.add(ParagraphStyle(name="H2Report", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=12,
                              leading=16, textColor=colors.HexColor("#1E293B"), spaceBefore=10, spaceAfter=5,
                              keepWithNext=True))
    styles.add(ParagraphStyle(name="BodyReport", parent=styles["BodyText"], fontSize=9.3, leading=13.2,
                              alignment=TA_LEFT, textColor=colors.HexColor("#243247"), spaceAfter=6))
    styles.add(ParagraphStyle(name="CodeReport", parent=styles["Code"], fontName="Courier", fontSize=7.8,
                              leading=10.2, backColor=colors.HexColor("#F1F5F9"), borderColor=colors.HexColor("#CBD5E1"),
                              borderWidth=0.4, borderPadding=6, spaceBefore=4, spaceAfter=8))
    styles.add(ParagraphStyle(name="BulletReport", parent=styles["BodyText"], fontSize=9.2, leading=13,
                              leftIndent=13, firstLineIndent=-9, textColor=colors.HexColor("#243247"), spaceAfter=3))
    return styles


def parse_table(lines, styles):
    cells = [[clean_inline(cell.strip()) for cell in row.strip().strip("|").split("|")] for row in lines if "---" not in row]
    formatted = [[Paragraph(cell, styles["BodyReport"]) for cell in row] for row in cells]
    table = Table(formatted, repeatRows=1, hAlign="LEFT", colWidths=[(A4[0] - 4 * cm) / len(cells[0])] * len(cells[0]))
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F766E")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F8FAFC")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def render():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    styles = make_styles()
    story = []
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    i = 0
    code_mode = False
    code_lines = []
    first_title = True
    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            if code_mode:
                story.append(Paragraph("<br/>".join(code_lines), styles["CodeReport"]))
                code_lines = []
            code_mode = not code_mode
            i += 1
            continue
        if code_mode:
            code_lines.append(line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
            i += 1
            continue
        if line.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].startswith("|"):
                table_lines.append(lines[i]); i += 1
            story.append(Spacer(1, 3)); story.append(parse_table(table_lines, styles)); story.append(Spacer(1, 7))
            continue
        if line.startswith("# "):
            if first_title:
                story += [Spacer(1, 5.3 * cm), Paragraph(clean_inline(line[2:]), styles["ReportTitle"]), Spacer(1, 8)]
                story.append(Paragraph("A text-AI design, implementation and evaluation plan", styles["Subtitle"]))
                story.append(Spacer(1, 4.5 * cm)); first_title = False
            else:
                story.append(Paragraph(clean_inline(line[2:]), styles["H1Report"]))
        elif line.startswith("## "):
            story.append(Paragraph(clean_inline(line[3:]), styles["H1Report"]))
        elif line.startswith("### "):
            story.append(Paragraph(clean_inline(line[4:]), styles["H2Report"]))
        elif re.match(r"^\d+\. ", line):
            story.append(Paragraph("• " + clean_inline(re.sub(r"^\d+\. ", "", line)), styles["BulletReport"]))
        elif line.startswith(("- ", "* ")):
            story.append(Paragraph("• " + clean_inline(line[2:]), styles["BulletReport"]))
        elif line.strip():
            story.append(Paragraph(clean_inline(line.rstrip("  ")), styles["BodyReport"]))
        else:
            story.append(Spacer(1, 3))
        i += 1
    doc = SimpleDocTemplate(str(OUTPUT), pagesize=A4, rightMargin=2 * cm, leftMargin=2 * cm,
                            topMargin=1.7 * cm, bottomMargin=2.1 * cm, title="SupportFlow AI Application Report")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUTPUT)


if __name__ == "__main__":
    render()
