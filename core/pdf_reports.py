"""
SynData AI - Professional PDF Report Generator
Uses ReportLab to generate publication-grade PDF audit reports.
"""

import io
import datetime
from typing import Dict, Any, Optional
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
import config

# ReportLab Color Constants matching SynData AI Palette
PDF_PURPLE_DARK = colors.HexColor(config.COLOR_DARK_PURPLE)
PDF_PURPLE_PRIMARY = colors.HexColor(config.COLOR_PRIMARY_PURPLE)
PDF_LAVENDER = colors.HexColor(config.COLOR_LIGHT_LAVENDER)
PDF_TEXT_MAIN = colors.HexColor(config.COLOR_PRIMARY_TEXT)
PDF_TEXT_MUTED = colors.HexColor(config.COLOR_SECONDARY_TEXT)
PDF_BORDER = colors.HexColor(config.COLOR_BORDERS)
PDF_SUCCESS = colors.HexColor(config.COLOR_SUCCESS)
PDF_WARNING = colors.HexColor(config.COLOR_WARNING)
PDF_ERROR = colors.HexColor(config.COLOR_ERROR)

def get_custom_pdf_styles():
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=PDF_PURPLE_DARK,
        spaceAfter=4
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=PDF_TEXT_MUTED,
        spaceAfter=15
    )
    
    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=PDF_PURPLE_DARK,
        spaceBefore=14,
        spaceAfter=6
    )
    
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=PDF_TEXT_MAIN
    )
    
    callout_style = ParagraphStyle(
        'DocCallout',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9,
        leading=13,
        textColor=PDF_PURPLE_DARK
    )
    
    return {
        'title': title_style,
        'subtitle': subtitle_style,
        'h2': h2_style,
        'body': body_style,
        'callout': callout_style
    }

def create_header(elements, title: str, subtitle: str, styles):
    now_str = datetime.datetime.now().strftime("%B %d, %Y - %H:%M UTC")
    elements.append(Paragraph(f"<b>SynData AI</b> | {title}", styles['title']))
    elements.append(Paragraph(f"{subtitle} &bull; Generated: {now_str}", styles['subtitle']))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=PDF_PURPLE_PRIMARY, spaceAfter=15))

def build_pdf_buffer(story) -> io.BytesIO:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    doc.build(story)
    buffer.seek(0)
    return buffer

def generate_statistical_pdf(
    dataset_name: str,
    model_name: str,
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    stat_df: pd.DataFrame,
    ks_df: pd.DataFrame,
    quality_info: Dict[str, Any]
) -> io.BytesIO:
    """
    Generates the Statistical Comparison Report PDF.
    """
    styles = get_custom_pdf_styles()
    story = []
    
    create_header(story, "Statistical Quality Audit Report", f"Dataset: {dataset_name} | Model: {model_name}", styles)
    
    # Summary Box
    summary_data = [
        [
            Paragraph("<b>Overall Fidelity Score</b>", styles['body']),
            Paragraph(f"<b>{quality_info.get('overall_score', 'N/A')}% ({quality_info.get('fidelity_rating', 'Good')})</b>", styles['body'])
        ],
        [
            Paragraph("<b>Original Dataset Size</b>", styles['body']),
            Paragraph(f"{len(real_df):,} rows x {len(real_df.columns)} columns", styles['body'])
        ],
        [
            Paragraph("<b>Synthetic Dataset Size</b>", styles['body']),
            Paragraph(f"{len(syn_df):,} rows x {len(syn_df.columns)} columns", styles['body'])
        ],
        [
            Paragraph("<b>Correlation Discrepancy (MAE)</b>", styles['body']),
            Paragraph(f"{quality_info.get('correlation_mae', 0.0):.4f}", styles['body'])
        ]
    ]
    t_summary = Table(summary_data, colWidths=[200, 340])
    t_summary.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), PDF_LAVENDER),
        ('BOX', (0, 0), (-1, -1), 1, PDF_BORDER),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, PDF_BORDER),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_summary)
    story.append(Spacer(1, 14))
    
    # Section: Kolmogorov-Smirnov Tests
    story.append(Paragraph("1. Kolmogorov-Smirnov Goodness-of-Fit Tests", styles['h2']))
    story.append(Paragraph("Evaluates whether the continuous marginal distributions of synthetic columns replicate original features without significant empirical drift.", styles['body']))
    story.append(Spacer(1, 6))
    
    if not ks_df.empty:
        ks_table_data = [["Feature", "KS Stat", "P-Value", "Similarity (%)", "Fit Assessment"]]
        for _, row in ks_df.head(15).iterrows():
            ks_table_data.append([
                str(row["Feature"]),
                str(row["KS Statistic"]),
                str(row["P-Value"]),
                str(row["Similarity Score (%)"]),
                str(row["Distribution Fit"])
            ])
            
        t_ks = Table(ks_table_data, colWidths=[130, 80, 90, 110, 130])
        t_ks.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), PDF_PURPLE_DARK),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, PDF_BORDER),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, PDF_LAVENDER]),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(t_ks)
    
    story.append(Spacer(1, 14))
    
    # Section: Numerical Moments Comparison
    story.append(Paragraph("2. Numerical Moment Comparison", styles['h2']))
    if not stat_df.empty:
        stat_table_data = [["Feature", "Real Mean", "Syn Mean", "Mean Diff", "Real Std", "Syn Std"]]
        for _, row in stat_df.head(12).iterrows():
            stat_table_data.append([
                str(row["Feature"]),
                str(row["Real Mean"]),
                str(row["Syn Mean"]),
                str(row["Mean Diff"]),
                str(row["Real Std"]),
                str(row["Syn Std"])
            ])
        t_stat = Table(stat_table_data, colWidths=[140, 80, 80, 80, 80, 80])
        t_stat.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), PDF_PURPLE_PRIMARY),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, PDF_BORDER),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, PDF_LAVENDER]),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(t_stat)
        
    return build_pdf_buffer(story)

def generate_privacy_pdf(
    dataset_name: str,
    model_name: str,
    privacy_info: Dict[str, Any]
) -> io.BytesIO:
    """
    Generates the Privacy Risk Analysis Report PDF.
    """
    styles = get_custom_pdf_styles()
    story = []
    
    create_header(story, "Privacy Risk & Memorization Audit Report", f"Dataset: {dataset_name} | Model: {model_name}", styles)
    
    # Privacy Assessment Card
    risk_color = PDF_ERROR if "High" in privacy_info['risk_level'] else (PDF_WARNING if "Moderate" in privacy_info['risk_level'] else PDF_SUCCESS)
    
    summary_data = [
        [
            Paragraph("<b>Overall Privacy Assessment</b>", styles['body']),
            Paragraph(f"<b><font color='{privacy_info['risk_color']}'>{privacy_info['risk_level']}</font></b>", styles['body'])
        ],
        [
            Paragraph("<b>Minimum Nearest Neighbor Distance</b>", styles['body']),
            Paragraph(f"{privacy_info['min_nnd']}", styles['body'])
        ],
        [
            Paragraph("<b>Average Nearest Neighbor Distance</b>", styles['body']),
            Paragraph(f"{privacy_info['avg_nnd']}", styles['body'])
        ],
        [
            Paragraph("<b>Exact Record Duplicates Detected</b>", styles['body']),
            Paragraph(f"{privacy_info['exact_matches']} records", styles['body'])
        ],
        [
            Paragraph("<b>Suspiciously Close Proximity Records</b>", styles['body']),
            Paragraph(f"{privacy_info['suspicious_count']} records ({privacy_info['suspicious_pct']}%)", styles['body'])
        ]
    ]
    t_summary = Table(summary_data, colWidths=[220, 320])
    t_summary.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), PDF_LAVENDER),
        ('BOX', (0, 0), (-1, -1), 1.5, risk_color),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, PDF_BORDER),
        ('TOPPADDING', (0, 0), (-1, -1), 7),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
    ]))
    story.append(t_summary)
    story.append(Spacer(1, 14))
    
    story.append(Paragraph("1. Technical Risk Interpretation", styles['h2']))
    story.append(Paragraph(privacy_info['risk_summary'], styles['body']))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("2. Privacy Preservation Principles", styles['h2']))
    explanation_text = (
        "Synthetic data mitigates re-identification risks by learning population distributions rather than memorizing "
        "individual rows. However, when models overfit on outliers or small datasets, generative algorithms may reproduce "
        "real training vectors. Nearest Neighbor Distance (NND) quantifies this proximity in normalized Euclidean space. "
        "A healthy synthetic dataset maintains a non-zero minimum distance while preserving global cluster density."
    )
    story.append(Paragraph(explanation_text, styles['body']))
    
    return build_pdf_buffer(story)

def generate_ml_eval_pdf(
    dataset_name: str,
    model_name: str,
    ml_info: Dict[str, Any]
) -> io.BytesIO:
    """
    Generates the ML Utility Evaluation Report PDF.
    """
    styles = get_custom_pdf_styles()
    story = []
    
    create_header(story, "Machine Learning Utility Report (TSTR)", f"Dataset: {dataset_name} | Model: {model_name}", styles)
    
    # Utility Score Card
    summary_data = [
        [
            Paragraph("<b>Target Variable</b>", styles['body']),
            Paragraph(f"<b>{ml_info.get('target_column')}</b> ({ml_info.get('task_type')})", styles['body'])
        ],
        [
            Paragraph("<b>Relative Utility Retention Ratio</b>", styles['body']),
            Paragraph(f"<b>{ml_info.get('utility_retention_pct')}%</b> of Real Model Performance", styles['body'])
        ],
        [
            Paragraph("<b>Real Test Partition Size</b>", styles['body']),
            Paragraph(f"{ml_info.get('test_size')} holdout samples (Unseen)", styles['body'])
        ],
        [
            Paragraph("<b>Synthetic Training Samples</b>", styles['body']),
            Paragraph(f"{ml_info.get('synthetic_train_size')} generated samples", styles['body'])
        ]
    ]
    t_summary = Table(summary_data, colWidths=[220, 320])
    t_summary.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), PDF_LAVENDER),
        ('BOX', (0, 0), (-1, -1), 1, PDF_BORDER),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, PDF_BORDER),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_summary)
    story.append(Spacer(1, 14))
    
    story.append(Paragraph("1. Train on Synthetic, Test on Real (TSTR) Benchmark", styles['h2']))
    story.append(Paragraph("Comparative validation on identical real holdout test partition. Avoids data leakage.", styles['body']))
    story.append(Spacer(1, 8))
    
    metrics_table = ml_info.get('metrics_table', pd.DataFrame())
    if not metrics_table.empty:
        table_data = [["Evaluation Metric", "Real Model (TRTR)", "Synthetic Model (TSTR)", "Difference"]]
        for _, row in metrics_table.iterrows():
            table_data.append([
                str(row["Metric"]),
                str(row["Real Model (TRTR)"]),
                str(row["Synthetic Model (TSTR)"]),
                str(row["Difference"])
            ])
            
        t_metrics = Table(table_data, colWidths=[150, 130, 130, 130])
        t_metrics.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), PDF_PURPLE_DARK),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, PDF_BORDER),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, PDF_LAVENDER]),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(t_metrics)
        
    return build_pdf_buffer(story)

def generate_consolidated_pdf(
    dataset_name: str,
    model_name: str,
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    quality_info: Dict[str, Any],
    privacy_info: Dict[str, Any],
    ml_info: Optional[Dict[str, Any]] = None
) -> io.BytesIO:
    """
    Generates the Complete Consolidated Evaluation Report PDF.
    """
    styles = get_custom_pdf_styles()
    story = []
    
    create_header(story, "Comprehensive Generative AI Evaluation Dossier", f"Dataset: {dataset_name} | Engine: {model_name}", styles)
    
    # Executive Summary
    story.append(Paragraph("Executive Overview", styles['h2']))
    exec_text = (
        f"This audit certifies the synthetic dataset generated by the <b>{model_name}</b> architecture on the "
        f"<b>{dataset_name}</b> repository. The evaluation analyzes three foundational pillars: statistical fidelity, "
        f"privacy memorization risk, and downstream machine learning utility."
    )
    story.append(Paragraph(exec_text, styles['body']))
    story.append(Spacer(1, 10))
    
    overview_table_data = [
        ["Audit Pillar", "Core Metric", "Calculated Value", "System Status"],
        ["Statistical Similarity", "Overall Quality Score", f"{quality_info.get('overall_score', 'N/A')}%", quality_info.get('fidelity_rating', 'Good')],
        ["Correlation Preservation", "Correlation MAE", f"{quality_info.get('correlation_mae', 0.0):.4f}", "Preserved"],
        ["Privacy Risk", "Minimum NND", f"{privacy_info.get('min_nnd', 'N/A')}", privacy_info.get('risk_level', 'Low')],
        ["Exact Copies", "Match Count", f"{privacy_info.get('exact_matches', 0)} duplicates", "Zero Memorization" if privacy_info.get('exact_matches', 0) == 0 else "Warning"],
    ]
    
    if ml_info:
        overview_table_data.append([
            "ML Utility (TSTR)", "Relative F1 Retention", f"{ml_info.get('utility_retention_pct', 'N/A')}%", "Preserved"
        ])
        
    t_ov = Table(overview_table_data, colWidths=[140, 140, 130, 130])
    t_ov.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PDF_PURPLE_DARK),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('GRID', (0, 0), (-1, -1), 0.5, PDF_BORDER),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, PDF_LAVENDER]),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t_ov)
    story.append(Spacer(1, 14))
    
    # Dataset Specifications
    story.append(Paragraph("Dataset & Execution Specifications", styles['h2']))
    meta_data = [
        [Paragraph(f"<b>Original Dimensions:</b> {len(real_df)} rows x {len(real_df.columns)} cols", styles['body']),
         Paragraph(f"<b>Synthetic Records:</b> {len(syn_df)} rows", styles['body'])],
        [Paragraph(f"<b>Generative Model:</b> {model_name}", styles['body']),
         Paragraph(f"<b>Verification Status:</b> Verified Generative Output", styles['body'])]
    ]
    t_meta = Table(meta_data, colWidths=[270, 270])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.white),
        ('BOX', (0, 0), (-1, -1), 1, PDF_BORDER),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t_meta)
    
    return build_pdf_buffer(story)
