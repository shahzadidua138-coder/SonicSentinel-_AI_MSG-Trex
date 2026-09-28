"""
src/services/pdf_generator.py - Official Acoustic Incident Certificate Generator
Generates forensic PDF reports with embedded SHA-256 fingerprint,
acoustic metrics, dual-AI confidence distribution, and incident audit log.
"""

import os
import io
from datetime import datetime


def generate_incident_pdf(event_data: dict) -> bytes:
    """
    Generates an official Acoustic Incident Certificate.
    Uses ReportLab if installed, otherwise constructs a standard PDF document.
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, Image
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
        story = []
        styles = getSampleStyleSheet()

        # Custom styles
        title_style = ParagraphStyle(
            'TitleStyle',
            parent=styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=18,
            leading=22,
            textColor=colors.HexColor('#032B43')
        )
        subtitle_style = ParagraphStyle(
            'SubTitleStyle',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=9,
            leading=12,
            textColor=colors.HexColor('#06B6D4')
        )
        body_style = ParagraphStyle(
            'Body',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=9,
            leading=12,
            textColor=colors.HexColor('#1E293B')
        )
        mono_style = ParagraphStyle(
            'Mono',
            parent=styles['Normal'],
            fontName='Courier',
            fontSize=8,
            leading=10,
            textColor=colors.HexColor('#0F172A')
        )

        # Header
        story.append(Paragraph("SONICSENTINEL AI — ACOUSTIC INCIDENT CERTIFICATE", title_style))
        story.append(Paragraph("NEXTWAVE AI & ML ACOUSTICX THREAT INTELLIGENCE PLATFORM", subtitle_style))
        story.append(Spacer(1, 10))
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#06B6D4'), spaceAfter=12))

        # Overview Table
        audio_id = event_data.get('id', 'AUD-UNKNOWN')
        filename = event_data.get('original_filename', 'audio_sample.wav')
        category = event_data.get('final_detected_class', event_data.get('threat_category', 'Unknown'))
        severity = event_data.get('severity', 'Unknown')
        quality = event_data.get('quality_grade', 'Unknown')
        snr = f"{event_data.get('snr_db', 'Unavailable')} dB"
        py_class = event_data.get('python_predicted_class', 'N/A')
        py_conf = f"{float(event_data.get('python_top_confidence', 0.0))*100:.1f}%"
        gtm_class = event_data.get('gtm_predicted_class', 'N/A')
        gtm_conf = f"{float(event_data.get('gtm_top_confidence', 0.0))*100:.1f}%"
        delta = f"{float(event_data.get('confidence_difference', 0.0))*100:.1f}%"
        status = event_data.get('consistency_status', 'Unavailable')
        sha = event_data.get('sha256_hash', 'Unavailable')

        meta_data = [
            [Paragraph("<b>Incident Token:</b>", body_style), Paragraph(audio_id, mono_style),
             Paragraph("<b>Acoustic Threat:</b>", body_style), Paragraph(f"<b>{category}</b>", body_style)],
            [Paragraph("<b>Source Recording:</b>", body_style), Paragraph(filename, mono_style),
             Paragraph("<b>Threat Severity:</b>", body_style), Paragraph(f"<font color='{'red' if severity=='Critical' else 'orange'}'><b>{severity}</b></font>", body_style)],
            [Paragraph("<b>Audio Quality:</b>", body_style), Paragraph(f"{quality} ({snr})", body_style),
             Paragraph("<b>Arbiter Status:</b>", body_style), Paragraph(status, body_style)],
            [Paragraph("<b>Timestamp:</b>", body_style), Paragraph(str(event_data.get('created_at', datetime.utcnow())), mono_style),
             Paragraph("<b>Incident Status:</b>", body_style), Paragraph(event_data.get('alert_status', 'Active'), body_style)]
        ]

        t_meta = Table(meta_data, colWidths=[100, 170, 100, 170])
        t_meta.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8FAFC')),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#E2E8F0')),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
            ('PADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(t_meta)
        story.append(Spacer(1, 14))

        # Dual AI Model Breakdown
        story.append(Paragraph("<b>INDEPENDENT MODEL COMPARISON</b>", subtitle_style))
        story.append(Spacer(1, 6))

        ai_data = [
            [Paragraph("<b>Inference Engine</b>", body_style), Paragraph("<b>Primary Detection</b>", body_style), Paragraph("<b>Confidence</b>", body_style), Paragraph("<b>Operational Role</b>", body_style)],
            [Paragraph("<b>Support Vector Machine</b>", body_style), Paragraph(py_class, body_style), Paragraph(py_conf, mono_style), Paragraph("Primary trained feature classifier", body_style)],
            [Paragraph("<b>Random Forest</b>", body_style), Paragraph(gtm_class, body_style), Paragraph(gtm_conf, mono_style), Paragraph("Independent trained comparison classifier", body_style)],
            [Paragraph("<b>Confidence Difference</b>", body_style), Paragraph(f"Delta: {delta}", mono_style), Paragraph(status, body_style), Paragraph("Model comparison status", body_style)]
        ]

        t_ai = Table(ai_data, colWidths=[160, 130, 90, 160])
        t_ai.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#032B43')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#CBD5E1')),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
            ('PADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(t_ai)
        story.append(Spacer(1, 14))

        # Source recording metadata
        story.append(Paragraph("<b>RECORDING METADATA</b>", subtitle_style))
        story.append(Spacer(1, 4))
        audio_meta = [
            [Paragraph("<b>Format</b>", body_style), Paragraph(str(event_data.get('audio_format') or 'Unknown'), body_style), Paragraph("<b>Duration</b>", body_style), Paragraph(f"{event_data.get('duration_seconds', 'Unknown')} s", body_style)],
            [Paragraph("<b>Sample Rate</b>", body_style), Paragraph(f"{event_data.get('sample_rate', 'Unknown')} Hz", body_style), Paragraph("<b>Channels</b>", body_style), Paragraph(str(event_data.get('channels', 'Unknown')), body_style)],
            [Paragraph("<b>Bit Depth</b>", body_style), Paragraph(str(event_data.get('bit_depth') or 'Unknown'), body_style), Paragraph("<b>File Size</b>", body_style), Paragraph(f"{event_data.get('file_size_bytes', 'Unknown')} bytes", body_style)],
        ]
        t_audio_meta = Table(audio_meta, colWidths=[80, 185, 80, 185])
        t_audio_meta.setStyle(TableStyle([('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#CBD5E1')), ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')), ('PADDING', (0, 0), (-1, -1), 6)]))
        story.append(t_audio_meta)
        story.append(Spacer(1, 12))

        project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        for image_field, heading in (("waveform_image_path", "WAVEFORM"), ("spectrogram_image_path", "SPECTROGRAM")):
            image_url = event_data.get(image_field)
            if image_url:
                image_path = os.path.join(project_dir, str(image_url).lstrip('/').replace('/', os.sep))
                if os.path.isfile(image_path):
                    story.append(Paragraph(f"<b>{heading}</b>", subtitle_style))
                    story.append(Spacer(1, 4))
                    story.append(Image(image_path, width=520, height=156))
                    story.append(Spacer(1, 8))

        # Recommended Action & SHA-256 Audit
        story.append(Paragraph("<b>IMMEDIATE DISPATCH & OPERATIONAL RECOMMENDATION</b>", subtitle_style))
        story.append(Spacer(1, 4))
        action_text = event_data.get('recommended_action', 'Inspect perimeter and notify security personnel.')
        story.append(Paragraph(action_text, body_style))
        story.append(Spacer(1, 10))

        story.append(Paragraph("<b>CRYPTOGRAPHIC AUDIT FINGERPRINT (SHA-256)</b>", subtitle_style))
        story.append(Spacer(1, 4))
        story.append(Paragraph(sha, mono_style))
        story.append(Spacer(1, 14))

        # Avoid certifying the incident as a real emergency response.
        sign_data = [
            [Paragraph(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}", body_style),
             Paragraph("Prototype analysis report - not an emergency-response certification.", body_style)]
        ]
        t_sign = Table(sign_data, colWidths=[270, 270])
        t_sign.setStyle(TableStyle([
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#06B6D4')),
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#ECFEFF')),
            ('PADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(t_sign)

        doc.build(story)
        return buffer.getvalue()
    except Exception:
        # Keep a valid, readable PDF even when the optional ReportLab renderer
        # is unavailable in a minimal installation.
        lines = [
            "SONICSENTINEL AI - ACOUSTIC ANALYSIS REPORT",
            f"Audio ID: {event_data.get('id', 'Unknown')}",
            f"File: {event_data.get('original_filename', 'Unknown')}",
            f"Detected category: {event_data.get('final_detected_class', 'Unknown')}",
            f"Severity / quality: {event_data.get('severity', 'Unknown')} / {event_data.get('quality_grade', 'Unknown')}",
            f"SVM: {event_data.get('python_predicted_class', 'Unknown')} ({float(event_data.get('python_top_confidence', 0.0))*100:.1f}%)",
            f"Random Forest: {event_data.get('gtm_predicted_class', 'Unknown')} ({float(event_data.get('gtm_top_confidence', 0.0))*100:.1f}%)",
            f"Confidence difference: {float(event_data.get('confidence_difference', 0.0))*100:.1f}%",
            f"Comparison status: {event_data.get('consistency_status', 'Unknown')}",
            f"Duration: {event_data.get('duration_seconds', 'Unknown')} seconds",
            f"Format / rate / channels: {event_data.get('audio_format', 'Unknown')} / {event_data.get('sample_rate', 'Unknown')} Hz / {event_data.get('channels', 'Unknown')}",
            f"SHA-256: {event_data.get('sha256_hash', 'Unavailable')}",
            "Prototype analysis report; not an emergency-response certification.",
        ]
        commands = ["BT /F1 14 Tf 48 748 Td"]
        for index, line in enumerate(lines):
            if index:
                commands.append("/F1 10 Tf 0 -28 Td")
            text = line.encode('latin-1', errors='replace').decode('latin-1')[:130]
            text = text.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')
            commands.append(f"({text}) Tj")
        commands.append("ET")
        stream = "\n".join(commands).encode('latin-1')
        objects = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            b"<< /Length " + str(len(stream)).encode('ascii') + b" >>\nstream\n" + stream + b"\nendstream",
        ]
        output = bytearray(b"%PDF-1.4\n")
        offsets = [0]
        for number, body in enumerate(objects, start=1):
            offsets.append(len(output))
            output.extend(f"{number} 0 obj\n".encode('ascii') + body + b"\nendobj\n")
        xref_offset = len(output)
        output.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode('ascii'))
        for offset in offsets[1:]:
            output.extend(f"{offset:010d} 00000 n \n".encode('ascii'))
        output.extend(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF".encode('ascii'))
        return bytes(output)
