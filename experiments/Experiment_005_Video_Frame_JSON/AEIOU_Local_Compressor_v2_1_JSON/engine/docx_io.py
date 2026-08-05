from __future__ import annotations

from datetime import datetime, timezone
from html import escape
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET


W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _node_text(node: ET.Element) -> str:
    parts: list[str] = []
    for child in node.iter():
        if child.tag == W + "t":
            parts.append(child.text or "")
        elif child.tag == W + "tab":
            parts.append("\t")
        elif child.tag in {W + "br", W + "cr"}:
            parts.append("\n")
    return "".join(parts)


def read_docx(path: Path) -> str:
    with zipfile.ZipFile(path, "r") as archive:
        xml_data = archive.read("word/document.xml")
    root = ET.fromstring(xml_data)
    body = root.find(W + "body")
    if body is None:
        return ""

    parts: list[str] = []
    for child in body:
        if child.tag == W + "p":
            parts.append(_node_text(child))
        elif child.tag == W + "tbl":
            rows: list[str] = []
            for row in child.findall(W + "tr"):
                cells = [
                    _node_text(cell)
                    for cell in row.findall(W + "tc")
                ]
                rows.append("\t".join(cells))
            parts.append("\n".join(rows))
    return "\n".join(parts)


def _paragraph_xml(text: str, *, bold: bool = False, size: int = 21, center: bool = False) -> str:
    paragraph_properties = (
        "<w:pPr><w:jc w:val=\"center\"/></w:pPr>"
        if center else ""
    )
    run_properties = [
        "<w:rFonts w:ascii=\"Microsoft YaHei\" w:eastAsia=\"Microsoft YaHei\" "
        "w:hAnsi=\"Microsoft YaHei\"/>",
        f"<w:sz w:val=\"{size}\"/><w:szCs w:val=\"{size}\"/>",
    ]
    if bold:
        run_properties.append("<w:b/><w:bCs/>")

    runs: list[str] = []
    lines = text.split("\n")
    for index, line in enumerate(lines):
        if index:
            runs.append("<w:r><w:br/></w:r>")
        runs.append(
            "<w:r><w:rPr>"
            + "".join(run_properties)
            + "</w:rPr><w:t xml:space=\"preserve\">"
            + escape(line)
            + "</w:t></w:r>"
        )
    return f"<w:p>{paragraph_properties}{''.join(runs)}</w:p>"


def write_docx(
    path: Path,
    *,
    title: str,
    subtitle: str,
    body_text: str,
) -> None:
    paragraphs = [
        _paragraph_xml(title, bold=True, size=34, center=True),
        _paragraph_xml(subtitle, size=18, center=True),
    ]
    for block in body_text.split("\n\n"):
        paragraphs.append(_paragraph_xml(block, size=21))

    document_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    %s
    <w:sectPr>
      <w:pgSz w:w="11906" w:h="16838"/>
      <w:pgMar w:top="850" w:right="1020" w:bottom="850" w:left="1020"
               w:header="400" w:footer="400" w:gutter="0"/>
    </w:sectPr>
  </w:body>
</w:document>
""" % "\n".join(paragraphs)

    styles_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:docDefaults>
    <w:rPrDefault>
      <w:rPr>
        <w:rFonts w:ascii="Microsoft YaHei" w:eastAsia="Microsoft YaHei" w:hAnsi="Microsoft YaHei"/>
        <w:sz w:val="21"/><w:szCs w:val="21"/>
      </w:rPr>
    </w:rPrDefault>
  </w:docDefaults>
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:spacing w:after="100" w:line="300" w:lineRule="auto"/></w:pPr>
  </w:style>
</w:styles>
"""

    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>
"""

    root_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
"""

    document_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>
"""

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    core_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties
 xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
 xmlns:dc="http://purl.org/dc/elements/1.1/"
 xmlns:dcterms="http://purl.org/dc/terms/"
 xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>{escape(title)}</dc:title>
  <dc:creator>AEIOU Local Compressor</dc:creator>
  <cp:lastModifiedBy>AEIOU Local Compressor</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>
"""

    app_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
 xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>AEIOU Local Compressor</Application>
</Properties>
"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", root_rels)
        archive.writestr("word/document.xml", document_xml)
        archive.writestr("word/styles.xml", styles_xml)
        archive.writestr("word/_rels/document.xml.rels", document_rels)
        archive.writestr("docProps/core.xml", core_xml)
        archive.writestr("docProps/app.xml", app_xml)
