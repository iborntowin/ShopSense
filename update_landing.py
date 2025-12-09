"""
Update landing.html forms to add proper IDs and result divs
"""

# Read the file
with open('templates/landing.html', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace segment form opening tag
content = content.replace(
    '<section class="form-section" id="preferences-form">\n    <div class="bd-container">\n        <h2 class="section-title">Segmentation des clients</h2>\n        <p class="section-subtitle">Identifiez le profil d\'un client selon ses habitudes d\'achat et ses préférences.</p>\n\n        <div class="form-container">\n            <form>',
    '<section class="form-section" id="preferences-form">\n    <div class="bd-container">\n        <h2 class="section-title">Segmentation des clients</h2>\n        <p class="section-subtitle">Analysez les produits populaires par segment de clientèle.</p>\n\n        <div class="form-container">\n            <form id="segmentForm">'
)

# Replace recommendation form opening tag  
content = content.replace(
    '<section class="form-section" id="prediction-form">\n    <div class="bd-container">\n        <h2 class="section-title">Recommandations personnalisées</h2>\n        <p class="section-subtitle">Analysez l\'historique d\'un client pour lui suggérer les produits les plus adaptés.</p>\n\n        <div class="form-container">\n            <form>',
    '<section class="form-section" id="prediction-form">\n    <div class="bd-container">\n        <h2 class="section-title">Recommandations personnalisées</h2>\n        <p class="section-subtitle">Obtenez des recommandations de produits pour un client spécifique.</p>\n\n        <div class="form-container">\n            <form id="recommendForm">'
)

# Add segment form fields - find and replace the entire form content for segment
segment_old_start = '''<form>
                <div class="form-group">
                    <label>Type de produits achetés</label>
                    <select>
                        <option>accessories</option>'''

segment_new_start = '''<form id="segmentForm">
                <div class="form-group">
                    <label>Segment ID (0-3)</label>
                    <select name="segment_id" id="segment_id" required>
                        <option value="0">Segment 0</option>
                        <option value="1">Segment 1</option>
                        <option value="2">Segment 2</option>
                        <option value="3">Segment 3</option>
                    </select>
                </div>

                <div class="form-group">
                    <label>Nombre de produits à afficher</label>
                    <input type="number" name="top_n" id="top_n" value="10" min="1" max="20">
                </div>

                <div id="segmentResult" style="display: none; margin-top: 20px; padding: 20px; background: #f0f8ff; border-radius: 10px;">
                    <h3>Résultats de segmentation</h3>
                    <div id="segmentContent"></div>
                </div>

                <div class="form-group" style="visibility:hidden">
                    <label>Placeholder</label>
                    <select>
                        <option>accessories</option>'''

content = content.replace(segment_old_start, segment_new_start)

# Add recommendation form fields
recommend_old_start = '''<form>
                <div class="form-group">
                    <label>Nombre de mois d'historique d'achat</label>
                    <input type="number" placeholder="Ex: 12">'''

recommend_new_start = '''<form id="recommendForm">
                <div class="form-group">
                    <label>ID Client (1-3900)</label>
                    <input type="number" name="customer_id" id="customer_id" placeholder="Ex: 100" value="100" min="1" max="3900" required>
                </div>

                <div class="form-group">
                    <label>Nombre de recommandations</label>
                    <input type="number" name="top_n_recommend" id="top_n_recommend" value="5" min="1" max="10">
                </div>

                <div id="recommendResult" style="display: none; margin-top: 20px; padding: 20px; background: #f0fff0; border-radius: 10px;">
                    <h3>Recommandations</h3>
                    <div id="recommendContent"></div>
                </div>

                <div class="form-group" style="visibility:hidden">
                    <label>Placeholder</label>
                    <input type="number" placeholder="Ex: 12">'''

content = content.replace(recommend_old_start, recommend_new_start)

# Write the modified content
with open('templates/landing.html', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ landing.html updated successfully!")
print("✅ Added form IDs and result display areas")
