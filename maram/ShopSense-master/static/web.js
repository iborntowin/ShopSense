/*==================== SHOW MENU ====================*/
const showMenu = (toggleId, navId) => {
    const toggle = document.getElementById(toggleId),
        nav = document.getElementById(navId);

    if (toggle && nav) {
        toggle.addEventListener('click', () => {
            nav.classList.toggle('show-menu');
        });
    }
};
showMenu('nav-toggle', 'nav-menu');

/*==================== REMOVE MENU MOBILE ====================*/
const navLink = document.querySelectorAll('.nav__link');

function linkAction() {
    const navMenu = document.getElementById('nav-menu');
    navMenu.classList.remove('show-menu');
}
navLink.forEach(n => n.addEventListener('click', linkAction));

/*==================== SCROLL SECTIONS ACTIVE LINK ====================*/
const sections = document.querySelectorAll('section[id]');

function scrollActive() {
    const scrollY = window.pageYOffset;

    sections.forEach(current => {
        const sectionHeight = current.offsetHeight;
        const sectionTop = current.offsetTop - 50;
        const sectionId = current.getAttribute('id');

        if (scrollY > sectionTop && scrollY <= sectionTop + sectionHeight) {
            document
                .querySelector('.nav__menu a[href*=' + sectionId + ']')
                .classList.add('active-link');
        } else {
            document
                .querySelector('.nav__menu a[href*=' + sectionId + ']')
                .classList.remove('active-link');
        }
    });
}
window.addEventListener('scroll', scrollActive);

/*==================== CHANGE BACKGROUND HEADER ====================*/
function scrollHeader() {
    const nav = document.getElementById('header');
    if (this.scrollY >= 80) nav.classList.add('scroll-header');
    else nav.classList.remove('scroll-header');
}
window.addEventListener('scroll', scrollHeader);

/*==================== SHOW SCROLL TOP ====================*/
function scrollTop() {
    const scrollTop = document.getElementById('scroll-top');
    if (this.scrollY >= 560) scrollTop.classList.add('show-scroll');
    else scrollTop.classList.remove('show-scroll');
}
window.addEventListener('scroll', scrollTop);

/*==================== DARK LIGHT THEME ====================*/
const themeButton = document.getElementById('theme-button');
const darkTheme = 'dark-theme';
const iconTheme = 'bx-sun';

// Previously selected topic (if user selected)
const selectedTheme = localStorage.getItem('selected-theme');
const selectedIcon = localStorage.getItem('selected-icon');

// Obtain the current theme
const getCurrentTheme = () =>
    document.body.classList.contains(darkTheme) ? 'dark' : 'light';
const getCurrentIcon = () =>
    themeButton.classList.contains(iconTheme) ? 'bx-moon' : 'bx-sun';

// Apply previously chosen theme
if (selectedTheme) {
    if (selectedTheme === 'dark') document.body.classList.add(darkTheme);
    if (selectedIcon === 'bx-moon') themeButton.classList.add(iconTheme);
}

// Toggle theme
themeButton.addEventListener('click', () => {
    document.body.classList.toggle(darkTheme);
    themeButton.classList.toggle(iconTheme);
    localStorage.setItem('selected-theme', getCurrentTheme());
    localStorage.setItem('selected-icon', getCurrentIcon());
});

/*==================== SCROLL REVEAL ANIMATION ====================*/
const sr = ScrollReveal({
    origin: 'top',
    distance: '30px',
    duration: 2000,
    reset: true,
});

sr.reveal(
    `.home__data, .about__data, .services__content, .footer__content`,
    {
        interval: 200,
    }
);

/*==================== SHOW / HIDE FORMS ====================*/
function showForm(formType) {
    const forms = document.querySelectorAll('.form-section');
    forms.forEach(f => f.classList.remove('active'));

    const form = document.getElementById(`${formType}-form`);
    if (form) {
        form.classList.add('active');
        const formContainer = form.querySelector('.form-container');
        if (formContainer) {
            formContainer.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    }
}

/*==================== CLEAR FORM ====================*/
function clearForm(button) {
    const form = button.closest('form');
    form.reset();
    
    // Hide result if exists
    const resultDiv = form.querySelector('.result-display');
    if (resultDiv) {
        resultDiv.style.display = 'none';
    }
}

/*==================== HANDLE SEGMENT FORM ====================*/
document.addEventListener('DOMContentLoaded', function() {
    // Customer Value Form Handler
    const customerValueForm = document.getElementById('customerValueForm');
    if (customerValueForm) {
        customerValueForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            // Show loading
            const submitBtn = customerValueForm.querySelector('button[type="submit"]');
            const originalText = submitBtn.textContent;
            submitBtn.textContent = 'Analyse en cours...';
            submitBtn.disabled = true;
            
            const formData = {
                age: parseInt(document.getElementById('cv_age').value),
                income: parseFloat(document.getElementById('cv_income').value),
                gender: document.getElementById('cv_gender').value,
                category: document.getElementById('cv_category').value,
                previous_purchases: parseInt(document.getElementById('cv_previous_purchases').value),
                review_rating: parseFloat(document.getElementById('cv_review_rating').value),
                num_web_visits: parseInt(document.getElementById('cv_num_web_visits').value),
                subscription: document.getElementById('cv_subscription').value,
                shipping: document.getElementById('cv_shipping').value,
                discount: document.getElementById('cv_discount').value,
                promo: document.getElementById('cv_promo').value
            };
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(formData)
                });
                
                const result = await response.json();
                
                if (result.error) {
                    document.getElementById('customerValueContent').innerHTML = `<p style="color: red;">Erreur: ${result.error}</p>`;
                    document.getElementById('customerValueResult').style.background = '#ffebee';
                } else {
                    // Set background color based on result
                    let backgroundColor = result.color === 'success' ? '#d4edda' : 
                                         result.color === 'warning' ? '#fff3cd' : '#f8d7da';
                    let textColor = result.color === 'success' ? '#155724' : 
                                   result.color === 'warning' ? '#856404' : '#721c24';
                    let badgeColor = result.color === 'success' ? '#28a745' : 
                                    result.color === 'warning' ? '#ffc107' : '#dc3545';
                    
                    document.getElementById('customerValueResult').style.background = backgroundColor;
                    document.getElementById('customerValueResult').style.color = textColor;
                    
                    let html = `
                        <div style="text-align: center; margin: 20px 0;">
                            <span style="font-size: 1.5em; padding: 10px 20px; background: ${badgeColor}; color: white; border-radius: 5px; display: inline-block;">${result.segment}</span>
                        </div>
                        <p><strong>Probabilité de haute valeur:</strong> ${result.probability_display}</p>
                        <h4 style="margin-top: 20px;">${result.recommendation.title}</h4>
                        <ul style="text-align: left; margin-top: 10px;">
                    `;
                    
                    result.recommendation.actions.forEach(action => {
                        html += `<li style="margin: 8px 0;">${action}</li>`;
                    });
                    
                    html += '</ul>';
                    
                    document.getElementById('customerValueContent').innerHTML = html;
                }
                document.getElementById('customerValueResult').style.display = 'block';
            } catch (error) {
                document.getElementById('customerValueContent').innerHTML = `<p style="color: red;">Erreur: ${error.message}</p>`;
                document.getElementById('customerValueResult').style.display = 'block';
            } finally {
                submitBtn.textContent = originalText;
                submitBtn.disabled = false;
            }
        });
    }
    
    // Segment Form Handler
    const segmentForm = document.getElementById('segmentForm');
    if (segmentForm) {
        segmentForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const segmentId = document.getElementById('segment_id').value;
            const topN = document.getElementById('top_n').value;
            
            try {
                const response = await fetch('/segment-info', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        segment_id: parseInt(segmentId),
                        top_n: parseInt(topN)
                    })
                });
                
                const result = await response.json();
                
                if (result.status === 'success') {
                    let html = `
                        <p><strong>Segment ${result.segment}</strong></p>
                        <p>Taille du segment: ${result.segment_size} clients</p>
                        <h4>Produits populaires:</h4>
                        <ol style="text-align: left;">
                    `;
                    
                    result.popular_items.forEach(item => {
                        html += `<li>${item.item} - ${item.purchases} achats (${(item.popularity * 100).toFixed(1)}%)</li>`;
                    });
                    
                    html += '</ol>';
                    
                    document.getElementById('segmentContent').innerHTML = html;
                    document.getElementById('segmentResult').style.display = 'block';
                } else {
                    document.getElementById('segmentContent').innerHTML = `<p style="color: red;">Erreur: ${result.error}</p>`;
                    document.getElementById('segmentResult').style.display = 'block';
                }
            } catch (error) {
                document.getElementById('segmentContent').innerHTML = `<p style="color: red;">Erreur: ${error.message}</p>`;
                document.getElementById('segmentResult').style.display = 'block';
            }
        });
    }
    
    const recommendForm = document.getElementById('recommendForm');
    if (recommendForm) {
        recommendForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const customerId = document.getElementById('customer_id').value;
            const topN = document.getElementById('top_n_recommend').value;
            
            try {
                const response = await fetch('/recommend', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        customer_id: parseInt(customerId),
                        top_n: parseInt(topN)
                    })
                });
                
                const result = await response.json();
                
                if (result.status === 'success') {
                    let html = `
                        <p><strong>Client ID:</strong> ${result.customer_id}</p>
                        <p><strong>Article acheté:</strong> ${result.purchased_item}</p>
                        <p><strong>Segment:</strong> ${result.segment}</p>
                        <h4>Recommandations:</h4>
                        <ol style="text-align: left;">
                    `;
                    
                    result.recommendations.forEach(rec => {
                        html += `<li>${rec.item} (Score: ${rec.score.toFixed(4)})</li>`;
                    });
                    
                    html += '</ol>';
                    
                    document.getElementById('recommendContent').innerHTML = html;
                    document.getElementById('recommendResult').style.display = 'block';
                } else {
                    document.getElementById('recommendContent').innerHTML = `<p style="color: red;">Erreur: ${result.error}</p>`;
                    document.getElementById('recommendResult').style.display = 'block';
                }
            } catch (error) {
                document.getElementById('recommendContent').innerHTML = `<p style="color: red;">Erreur: ${error.message}</p>`;
                document.getElementById('recommendResult').style.display = 'block';
            }
        });
    }
});
