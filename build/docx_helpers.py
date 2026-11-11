"""Small python-docx helpers for the group report (A4, compact academic layout)."""
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
import re

INK = RGBColor(0x0B, 0x0B, 0x0B)
GREY = RGBColor(0x52, 0x51, 0x4E)
BLUE = RGBColor(0x1C, 0x5C, 0xAB)
FONT = "Calibri"

def new_document(body_pt=10, margins_cm=2.0):
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(margins_cm)
    sec.top_margin, sec.bottom_margin = Cm(1.9), Cm(1.8)
    st = doc.styles["Normal"]
    st.font.name = FONT; st.font.size = Pt(body_pt); st.font.color.rgb = INK
    st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    pf = st.paragraph_format
    pf.space_before = Pt(0); pf.space_after = Pt(4.5); pf.line_spacing = 1.07
    for name, size, before, after, color in [("Heading 1", 13.5, 10, 3, BLUE), ("Heading 2", 11, 7, 2, INK), ("Heading 3", 10, 5, 1, GREY)]:
        h = doc.styles[name]
        h.font.name = FONT; h.font.size = Pt(size); h.font.bold = True; h.font.color.rgb = color
        h.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
        h.element.rPr.rFonts.set(qn("w:ascii"), FONT); h.element.rPr.rFonts.set(qn("w:hAnsi"), FONT)
        h.paragraph_format.space_before = Pt(before); h.paragraph_format.space_after = Pt(after)
        h.paragraph_format.keep_with_next = True
    return doc

def _add_runs(p, text, size=None, color=None, italic=False):
    """Supports **bold** and *italic* inline markup."""
    for part in re.split(r"(\*\*.+?\*\*|`.+?`|\*[^*\s][^*]*?\*)", text):
        if not part:
            continue
        if part.startswith("**"):
            r = p.add_run(part[2:-2]); r.bold = True
        elif part.startswith("`"):
            r = p.add_run(part[1:-1]); r.font.name = "Consolas"
            r._element.rPr.rFonts.set(qn("w:eastAsia"), "Consolas")
            if size: r.font.size = Pt(size - 0.5)
            else: r.font.size = Pt(9)
            if color is not None: r.font.color.rgb = color
            continue
        elif part.startswith("*"):
            r = p.add_run(part[1:-1]); r.italic = True
        else:
            r = p.add_run(part)
        if italic: r.italic = True
        if size: r.font.size = Pt(size)
        if color is not None: r.font.color.rgb = color

def para(doc, text, size=None, align=None, after=None, before=None, color=None, italic=False, keep_next=False):
    p = doc.add_paragraph()
    _add_runs(p, text, size, color, italic)
    if align == "justify": p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    elif align == "center": p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if after is not None: p.paragraph_format.space_after = Pt(after)
    if before is not None: p.paragraph_format.space_before = Pt(before)
    if keep_next: p.paragraph_format.keep_with_next = True
    return p

def bullet(doc, text, size=None, after=2):
    p = doc.add_paragraph(style="List Bullet")
    _add_runs(p, text, size)
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.left_indent = Cm(0.55); p.paragraph_format.first_line_indent = Cm(-0.4)
    return p

def heading(doc, text, level=1):
    return doc.add_heading(text, level=level)

def figure(doc, path, width_cm, caption, note=None):
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(1); p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(path), width=Cm(width_cm))
    c = doc.add_paragraph()
    _add_runs(c, caption, size=8.5, color=GREY)
    c.paragraph_format.space_after = Pt(6)
    return c

def _shade(cell, hex_fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd"); shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), hex_fill)
    tcPr.append(shd)

def _cell_borders(cell, **kw):
    tcPr = cell._tc.get_or_add_tcPr()
    b = OxmlElement("w:tcBorders")
    for edge in ("top", "left", "bottom", "right"):
        v = kw.get(edge)
        el = OxmlElement(f"w:{edge}")
        if v:
            el.set(qn("w:val"), "single"); el.set(qn("w:sz"), str(v[0])); el.set(qn("w:color"), v[1])
        else:
            el.set(qn("w:val"), "nil")
        b.append(el)
    tcPr.append(b)

def table(doc, rows, col_widths_cm, header=True, size=8.5, align=None, bold_first_col=False, highlight_rows=()):
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    tblPr = t._tbl.tblPr
    lay = OxmlElement("w:tblLayout"); lay.set(qn("w:type"), "fixed"); tblPr.append(lay)
    for j, wcm in enumerate(col_widths_cm):
        t.columns[j].width = Cm(wcm)
    grid = t._tbl.tblGrid
    for j, gc in enumerate(grid.findall(qn("w:gridCol"))):
        gc.set(qn("w:w"), str(int(col_widths_cm[j] * 567)))
    for row in t.rows:                       # never split a row across pages
        trPr = row._tr.get_or_add_trPr()
        cs = OxmlElement("w:cantSplit"); trPr.append(cs)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = t.cell(i, j)
            c.width = Cm(col_widths_cm[j])
            c.text = ""
            p = c.paragraphs[0]
            p.paragraph_format.space_after = Pt(1); p.paragraph_format.space_before = Pt(1); p.paragraph_format.line_spacing = 1.0
            if i < len(rows) - 1: p.paragraph_format.keep_with_next = True      # keep the table on one page
            _add_runs(p, str(val), size=size)
            if align and align[j] == "r": p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            elif align and align[j] == "c": p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            if header and i == 0:
                _shade(c, "1C5CAB")
                for r in p.runs: r.bold = True; r.font.color.rgb = RGBColor(255, 255, 255)
                _cell_borders(c)
            else:
                if i in highlight_rows:
                    _shade(c, "FCE9E1")
                    for r in p.runs: r.bold = True
                _cell_borders(c, bottom=(4, "E1E0D9"))
            if bold_first_col and j == 0 and i > 0:
                for r in p.runs: r.bold = True
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t

def page_break(doc):
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

def add_page_number_footer(doc, left_text):
    sec = doc.sections[0]
    p = sec.footer.paragraphs[0]
    p.text = ""
    r = p.add_run(left_text + "    |    Page "); r.font.size = Pt(8); r.font.color.rgb = GREY
    for instr in ("begin", None, "end"):
        run = p.add_run(); run.font.size = Pt(8); run.font.color.rgb = GREY
        if instr == "begin":
            f = OxmlElement("w:fldChar"); f.set(qn("w:fldCharType"), "begin"); run._r.append(f)
        elif instr is None:
            it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = "PAGE"; run._r.append(it)
        else:
            f = OxmlElement("w:fldChar"); f.set(qn("w:fldCharType"), "end"); run._r.append(f)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
