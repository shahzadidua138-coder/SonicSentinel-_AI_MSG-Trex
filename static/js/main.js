/**
 * SonicSentinel AI - Enterprise Frontend Client Script
 */

document.addEventListener('DOMContentLoaded', () => {
    // Unified NextWave Acoustic Light Theme Enforcer
    try {
        localStorage.removeItem('sonic_theme');
        localStorage.removeItem('sonicsentinel_theme');
    } catch(e) {}
    document.documentElement.setAttribute('data-theme', 'light');
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
