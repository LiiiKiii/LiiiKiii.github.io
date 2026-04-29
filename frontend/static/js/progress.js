
document.addEventListener('DOMContentLoaded', function() {
  if (typeof loadPreferences === 'function') {
    loadPreferences();
  }
  
  if (typeof setupNavigation === 'function') {
    setupNavigation();
  }
  
  initScrollFadeIn();
});

function initScrollFadeIn() {
  const fadeElements = document.querySelectorAll('.scroll-fade-in');
  
  if (fadeElements.length === 0) {
    return;
  }
  
  const observerOptions = {
    root: null,
    rootMargin: '0px 0px -15% 0px', // Trigger 15% earlier
    threshold: 0.1 // Only check whether the element enters the viewport
  };
  
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const element = entry.target;
        element.classList.add('visible');
        observer.unobserve(element);
      }
    });
  }, observerOptions);
  
  fadeElements.forEach(element => {
    observer.observe(element);
  });
  
  let scrollTimeout = null;
  
  function checkBottom() {
    const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
    const scrollHeight = document.documentElement.scrollHeight;
    const clientHeight = document.documentElement.clientHeight;
    const isAtBottom = scrollTop + clientHeight >= scrollHeight - 100;
    
    if (isAtBottom) {
      const progressModules = document.querySelector('.progress-modules');
      if (progressModules) {
        const moduleElements = progressModules.querySelectorAll('.scroll-fade-in');
        moduleElements.forEach(element => {
          element.classList.add('visible');
        });
        progressModules.classList.add('visible');
      }
    }
  }
  
  window.addEventListener('scroll', () => {
    if (scrollTimeout) {
      clearTimeout(scrollTimeout);
    }
    scrollTimeout = setTimeout(checkBottom, 100);
  }, { passive: true });
  
  setTimeout(checkBottom, 200);
}
