from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from io import BytesIO
import datetime

def generate_itinerary_pdf(location, days, start_date, itinerary_data, hotels):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50)
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor("#EA580C"),  # Orange-600
        alignment=1,
        spaceAfter=20
    )
    
    day_header_style = ParagraphStyle(
        'DayHeaderStyle',
        parent=styles['Heading2'],
        fontSize=18,
        textColor=colors.HexColor("#1F2937"),  # Gray-800
        spaceBefore=15,
        spaceAfter=10,
        underlineWidth=1,
        borderPadding=5
    )
    
    place_name_style = ParagraphStyle(
        'PlaceNameStyle',
        parent=styles['Normal'],
        fontSize=12,
        fontWeight='bold',
        textColor=colors.HexColor("#1F2937")
    )

    hotel_style = ParagraphStyle(
        'HotelStyle',
        parent=styles['Normal'],
        fontSize=12,
        textColor=colors.HexColor("#1F2937"),
        spaceAfter=5
    )

    elements = []

    # Title
    elements.append(Paragraph(f"AI Travel Planner – {location}", title_style))
    elements.append(Paragraph(f"Duration: {days} Days | Start Date: {start_date}", styles['Normal']))
    elements.append(Spacer(1, 20))

    # Itinerary Plan
    plan = itinerary_data.get("plan", {})
    transport = itinerary_data.get("transport", {})

    # Arrival Info
    elements.append(Paragraph("Travel Logistics", styles['Heading2']))
    elements.append(Paragraph(f"<b>Nearest Airport:</b> {transport.get('airport', 'Not Available')}", styles['Normal']))
    elements.append(Paragraph(f"<b>Major Railway Station:</b> {transport.get('railway', 'Not Available')}", styles['Normal']))
    elements.append(Spacer(1, 20))

    for day_label, activities in plan.items():
        elements.append(Paragraph(day_label, day_header_style))
        
        daily_cost = 0
        
        for activity in activities:
            if isinstance(activity, str):
                if "Estimated Cost" in activity:
                    try:
                        cost_val = "".join(filter(str.isdigit, activity))
                        if cost_val: daily_cost = int(cost_val)
                    except:
                        pass
                else:
                    elements.append(Paragraph(f"• {activity}", styles['Normal']))
                continue
            
            # Place details
            p_name = activity.get("place_name", "Unknown Place")
            p_rating = activity.get("rating", "N/A")
            p_time = activity.get("visit_time", "N/A")
            p_desc = activity.get("short_description", "")
            
            elements.append(Paragraph(f"<b>{p_name}</b> – Rating: {p_rating} – Visit Time: {p_time} hrs", styles['Normal']))
            if p_desc:
                elements.append(Paragraph(f"<i>{p_desc}</i>", ParagraphStyle('Italic', parent=styles['Normal'], leftIndent=10, fontSize=10)))
            elements.append(Spacer(1, 5))

        elements.append(Spacer(1, 5))
        elements.append(Paragraph(f"<b>Estimated Daily Cost:</b> ₹{daily_cost}", styles['Normal']))
        elements.append(Spacer(1, 15))

    # Recommended Hotels
    if hotels:
        elements.append(PageBreak())
        elements.append(Paragraph("Recommended Hotels", styles['Heading2']))
        
        hotel_data = [["Hotel Name", "Price (per night)", "Rating"]]
        for hotel in hotels:
            hotel_data.append([
                hotel.get("name", "N/A"),
                f"₹{hotel.get('price', 'N/A')}",
                str(hotel.get("rating", "N/A"))
            ])
        
        t = Table(hotel_data, colWidths=[250, 120, 80])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#FFF7ED")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor("#EA580C")),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey)
        ]))
        elements.append(t)

    # Footer/Travel Tips
    elements.append(Spacer(1, 30))
    elements.append(Paragraph("Travel Tips", styles['Heading3']))
    tips = [
        "Carry light clothing and comfortable walking shoes.",
        "Keep digital copies of your ID and itinerary.",
        "Check local weather forecasts before heading out.",
        "Respect local customs and traditions."
    ]
    for tip in tips:
        elements.append(Paragraph(f"• {tip}", styles['Normal']))

    doc.build(elements)
    buffer.seek(0)
    return buffer
