/**
 * SonicSentinel AI - Enterprise Frontend Client Script
 */

document.addEventListener('DOMContentLoaded', () => {
    // Theme Initializer - Default to Light Theme per User Specification
    const savedTheme = localStorage.getItem('sonic_theme') || 'light';
    document.documentElement.setAttribute('data-theme', savedTheme);
    updateThemeIcon(savedTheme);

    const themeToggleBtn = document.getElementById('theme-toggle-btn');
    if (themeToggleBtn) {
        themeToggleBtn.addEventListener('click', () => {
            const currentTheme = document.documentElement.getAttribute('data-theme') || 'light';
            const newTheme = (currentTheme === 'light') ? 'dark' : 'light';
            document.documentElement.setAttribute('data-theme', newTheme);
            localStorage.setItem('sonic_theme', newTheme);
            updateThemeIcon(newTheme);
        });
    }

    function updateThemeIcon(theme) {
        const icon = document.querySelector('#theme-toggle-btn i');
        if (icon) {
            icon.className = (theme === 'dark') ? 'fas fa-sun' : 'fas fa-moon';
            themeToggleBtn.title = (theme === 'dark') ? 'Switch to Light Theme' : 'Switch to Dark Theme';
        }
    }
});

// Toast notification helper with Dark Teal Green styling
function showToast(msg, type = 'info') {
    let container = document.getElementById('toast-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'toast-container';
        container.style.cssText = 'position:fixed;bottom:24px;right:24px;z-index:9999;display:flex;flex-direction:column;gap:12px;';
        document.body.appendChild(container);
    }

    const toast = document.createElement('div');
    toast.className = `flash-alert ${type}`;
    toast.style.cssText = 'min-width:300px;box-shadow:0 8px 30px rgba(6,47,44,0.18);border-radius:10px;padding:14px 18px;background:#062F2C;color:#FFFFFF;border:1.5px solid rgba(45,212,191,0.35);font-size:13px;display:flex;align-items:center;gap:10px;animation:slideInRight 0.3s ease;';
    toast.innerHTML = `<i class="fas fa-info-circle" style="color:#2DD4BF;"></i><span>${msg}</span>`;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateY(10px)';
        toast.style.transition = 'all 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}
