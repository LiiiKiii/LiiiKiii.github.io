
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
  const contactSection = document.querySelector('.contact-section');
  const fadeElements = document.querySelectorAll('.scroll-fade-in');
  const allElements = contactSection ? [contactSection, ...fadeElements] : [...fadeElements];
  
  if (allElements.length === 0) {
    return;
  }
  
  setTimeout(() => {
    checkVisibility();
  }, 100);
  
  let ticking = false;
  function handleScroll() {
    if (!ticking) {
      window.requestAnimationFrame(() => {
        checkVisibility();
        ticking = false;
      });
      ticking = true;
    }
  }
  
  window.addEventListener('scroll', handleScroll, { passive: true });
  
  function checkVisibility() {
    const windowHeight = window.innerHeight;
    const triggerPoint = windowHeight * 0.8;
    const fadeInRange = windowHeight * 0.6;
    
    allElements.forEach((element) => {
      const rect = element.getBoundingClientRect();
      const elementTop = rect.top;
      const elementHeight = rect.height;
      const elementBottom = rect.bottom;
      
      if (elementTop < triggerPoint && elementTop > -elementHeight) {
        const progress = Math.max(0, Math.min(1, (triggerPoint - elementTop) / fadeInRange));
        element.style.opacity = progress.toString();
        element.style.transform = `translateY(${50 * (1 - progress)}px)`;
        
        if (progress >= 0.9) {
          element.classList.add('visible');
        }
      } else if (elementTop <= -elementHeight) {
        element.style.opacity = '1';
        element.style.transform = 'translateY(0)';
        element.classList.add('visible');
      } else if (elementTop < windowHeight && elementBottom > 0) {
        const progress = Math.max(0, Math.min(1, (windowHeight - elementTop) / fadeInRange));
        element.style.opacity = progress.toString();
        element.style.transform = `translateY(${50 * (1 - progress)}px)`;
        
        if (progress >= 0.9) {
          element.classList.add('visible');
        }
      }
    });
  }
  
  checkVisibility();
}
