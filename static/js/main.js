/**
 * Street Vendor Digitalization Agent — main.js
 * Global JS utilities loaded on every page.
 */

document.addEventListener('DOMContentLoaded', () => {

  // ── Active nav link highlight ──
  const currentPath = window.location.pathname;
  document.querySelectorAll('#mainNav .nav-link').forEach(link => {
    if (link.getAttribute('href') === currentPath) {
      link.classList.add('active');
    }
  });

  // ── Smooth scroll for anchor links ──
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
      const target = document.querySelector(this.getAttribute('href'));
      if (target) {
        e.preventDefault();
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  });

  // ── Auto-dismiss alerts after 5 seconds ──
  document.querySelectorAll('.alert.alert-dismissible').forEach(alert => {
    setTimeout(() => {
      const bsAlert = bootstrap.Alert.getOrCreateInstance(alert);
      bsAlert.close();
    }, 5000);
  });

  // ── Textarea auto-resize ──
  document.querySelectorAll('textarea.auto-resize').forEach(ta => {
    ta.addEventListener('input', function () {
      this.style.height = 'auto';
      this.style.height = (this.scrollHeight) + 'px';
    });
  });

  // ── Copy-to-clipboard utility (global) ──
  window.copyText = function (text) {
    navigator.clipboard.writeText(text).then(() => {
      showGlobalToast('Copied!', 'success');
    }).catch(() => {
      showGlobalToast('Copy failed — please copy manually.', 'warning');
    });
  };

  // ── Global toast utility ──
  window.showGlobalToast = function (message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `alert alert-${type} position-fixed`;
    toast.style.cssText = 'bottom:1.5rem;right:1.5rem;z-index:9999;min-width:220px;box-shadow:0 4px 20px rgba(0,0,0,.15)';
    toast.innerHTML = message;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 3000);
  };

  // ── Print-safe: hide nav & footer for print ──
  window.addEventListener('beforeprint', () => {
    document.querySelector('#mainNav')?.classList.add('d-none');
    document.querySelector('.footer')?.classList.add('d-none');
  });
  window.addEventListener('afterprint', () => {
    document.querySelector('#mainNav')?.classList.remove('d-none');
    document.querySelector('.footer')?.classList.remove('d-none');
  });

  // ── Language preference persistence ──
  const langSelect = document.getElementById('langSelect');
  if (langSelect) {
    const saved = localStorage.getItem('vendorai_language');
    if (saved) langSelect.value = saved;
    langSelect.addEventListener('change', () => {
      localStorage.setItem('vendorai_language', langSelect.value);
    });
  }

});
