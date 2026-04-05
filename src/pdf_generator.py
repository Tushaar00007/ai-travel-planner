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
            
            # Transport Fares Block
            transport_fares = activity.get("transport_fares")
            if transport_fares:
                def _fare_cell(val):
                    return "❌ Not Available in this city" if str(val).strip() == "Not Available" else val

                fare_data = [
                    ["Vehicle", "Fare Info"],
                    ["Auto Rickshaw", transport_fares.get("auto", "N/A")],
                    ["Rapido Bike", transport_fares.get("rapido_bike", "N/A")],
                    ["Ola Car", _fare_cell(transport_fares.get("ola_car", "N/A"))],
                    ["Uber Car", _fare_cell(transport_fares.get("uber_car", "N/A"))],
                ]
                fare_table = Table(fare_data, colWidths=[150, 250])
                fare_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#FFF7ED")),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor("#EA580C")),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#FFF7ED")])
                ]))
                elements.append(fare_table)
                if transport_fares.get("surge_note") and transport_fares.get("surge_note") != "N/A":
                    elements.append(Paragraph(f"<i>⚡ {transport_fares['surge_note']}</i>", ParagraphStyle('small', parent=styles['Normal'], fontSize=8, textColor=colors.grey)))
                if transport_fares.get("ola_car") == "Not Available" or transport_fares.get("uber_car") == "Not Available":
                    elements.append(Paragraph(
                        "<i>* Ola/Uber may not be available in smaller or remote cities.</i>",
                        ParagraphStyle('OlaUberNote', parent=styles['Normal'], fontSize=8, textColor=colors.HexColor("#EA580C"))
                    ))
                elements.append(Spacer(1, 5))

            # Nightlife Info Block
            nightlife = activity.get("nightlife_info")
            if nightlife:
                nl_data = [
                    ["Nightlife Info", ""],
                    ["Venue Type", nightlife.get("venue_type", "N/A")],
                    ["Cover Charge", nightlife.get("cover_charge", "N/A")],
                    ["Avg Drinks", nightlife.get("avg_drinks_price", "N/A")],
                    ["Music", nightlife.get("music_genre", "N/A")],
                    ["Timings", nightlife.get("club_timings", "N/A")],
                    ["Dress Code", nightlife.get("dress_code", "N/A")],
                    ["Ladies Night", nightlife.get("ladies_night", "N/A")],
                ]
                nl_table = Table(nl_data, colWidths=[150, 250])
                nl_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1F2937")),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('SPAN', (0, 0), (-1, 0)),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F4F6")])
                ]))
                elements.append(nl_table)
                elements.append(Spacer(1, 5))
            
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
