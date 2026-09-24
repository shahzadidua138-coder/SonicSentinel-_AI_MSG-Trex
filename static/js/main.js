/**
 * SonicSentinel AI - Enterprise Frontend Client Script
 */

document.addEventListener('DOMContentLoaded', () => {
    // Theme Initializer
    const savedTheme = localStorage.getItem('sonic_theme') || 'dark';
    document.documentElement.setAttribute('data-theme', savedTheme);
    updateThemeIcon(savedTheme);

    const themeToggleBtn = document.getElementById('theme-toggle-btn');
    if (themeToggleBtn) {
        themeToggleBtn.addEventListener('click', () => {
            const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
            const newTheme = (currentTheme === 'dark') ? 'light' : 'dark';
            document.documentElement.setAttribute('data-theme', newTheme);
            localStorage.setItem('sonic_theme', newTheme);
            updateThemeIcon(newTheme);
        });
    }

    function updateThemeIcon(theme) {
        const icon = document.querySelector('#theme-toggle-btn i');
        if (icon) {
            icon.className = (theme === 'dark') ? 'fas fa-sun' : 'fas fa-moon';
        }
    }
});

// Toast notification helper
function showToast(msg, type = 'info') {
    let container = document.getElementById('toast-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'toast-container';
        container.style.cssText = 'position:fixed;bottom:20px;right:20px;z-index:9999;display:flex;flex-direction:column;gap:10px;';
        document.body.appendChild(container);
    }

    const toast = document.createElement('div');
    toast.className = `flash-alert ${type}`;
    toast.style.cssText = 'min-width:280px;box-shadow:0 6px 20px rgba(0,0,0,0.4);border-radius:6px;padding:12px 16px;background:#111827;color:#fff;border:1px solid #374151;';
    toast.innerHTML = `<span>${msg}</span>`;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transition = 'opacity 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}
