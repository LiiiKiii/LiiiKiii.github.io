
document.addEventListener('DOMContentLoaded', function() {
  if (typeof loadPreferences === 'function') {
    loadPreferences();
  }
  if (typeof setupNavigation === 'function') {
    setupNavigation();
  }
  
  initAPIKeyManagement();
  
  initScrollFadeIn();
  
  window.scrollTo(0, 0);
});

function initAPIKeyManagement() {
  // OpenAI API Key
  const openaiInput = document.getElementById('openai-key');
  const openaiToggle = document.getElementById('openai-toggle');
  const openaiSave = document.getElementById('openai-save');
  const openaiDelete = document.getElementById('openai-delete');
  const openaiStatus = document.getElementById('openai-status');
  
  loadAPIKey('openai', openaiInput, openaiStatus, openaiDelete);
  
  if (openaiToggle) {
    openaiToggle.addEventListener('click', () => {
      togglePasswordVisibility(openaiInput, openaiToggle);
    });
  }
  
  if (openaiSave) {
    openaiSave.addEventListener('click', () => {
      saveAPIKey('openai', openaiInput.value.trim(), openaiStatus, openaiDelete, openaiInput);
    });
  }
  
  if (openaiDelete) {
    openaiDelete.addEventListener('click', () => {
      deleteAPIKey('openai', openaiInput, openaiStatus, openaiDelete);
    });
  }
  
  if (openaiInput) {
    openaiInput.addEventListener('keypress', (e) => {
      if (e.key === 'Enter') {
        saveAPIKey('openai', openaiInput.value.trim(), openaiStatus, openaiDelete, openaiInput);
      }
    });
  }
}

function loadAPIKey(keyName, inputElement, statusElement, deleteButton) {
  const savedKey = localStorage.getItem(`api_key_${keyName}`);
  if (savedKey) {
    inputElement.value = savedKey;
    updateStatus(statusElement, true);
    deleteButton.style.display = 'block';
    inputElement.type = 'password';
  } else {
    updateStatus(statusElement, false);
    deleteButton.style.display = 'none';
  }
}

function saveAPIKey(keyName, keyValue, statusElement, deleteButton, inputElement) {
  if (!keyValue) {
    alert('Please enter an API key.');
    return;
  }
  
  if (keyName === 'openai' && !keyValue.startsWith('sk-')) {
    if (!confirm('OpenAI API keys usually start with "sk-". Do you want to save it anyway?')) {
      return;
    }
  }
  
  localStorage.setItem(`api_key_${keyName}`, keyValue);
  
  updateStatus(statusElement, true);
  deleteButton.style.display = 'block';
  inputElement.type = 'password';
  
  showMessage('API key saved.', 'success');
}

function deleteAPIKey(keyName, inputElement, statusElement, deleteButton) {
  if (!confirm('Are you sure you want to delete the saved API key?')) {
    return;
  }
  
  localStorage.removeItem(`api_key_${keyName}`);
  
  inputElement.value = '';
  updateStatus(statusElement, false);
  deleteButton.style.display = 'none';
  inputElement.type = 'password';
  
  showMessage('API key deleted.', 'info');
}

function updateStatus(statusElement, isSaved) {
  if (isSaved) {
    statusElement.textContent = 'Saved';
    statusElement.classList.add('saved');
  } else {
    statusElement.textContent = 'Not set';
    statusElement.classList.remove('saved');
  }
}

function togglePasswordVisibility(inputElement, toggleButton) {
  const isPassword = inputElement.type === 'password';
  inputElement.type = isPassword ? 'text' : 'password';
  
  const svg = toggleButton.querySelector('svg');
  if (isPassword) {
    svg.innerHTML = `
      <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"></path>
      <line x1="1" y1="1" x2="23" y2="23"></line>
    `;
  } else {
    svg.innerHTML = `
      <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path>
      <circle cx="12" cy="12" r="3"></circle>
    `;
  }
}

function showMessage(message, type) {
  const messageEl = document.createElement('div');
  messageEl.className = `api-message api-message-${type}`;
  messageEl.textContent = message;
  
  const container = document.querySelector('.ai-enhance-container');
  if (container) {
    container.appendChild(messageEl);
    
    setTimeout(() => {
      messageEl.style.opacity = '1';
      messageEl.style.transform = 'translateY(0)';
    }, 10);
    
    setTimeout(() => {
      messageEl.style.opacity = '0';
      messageEl.style.transform = 'translateY(-10px)';
      setTimeout(() => {
        if (messageEl.parentNode) {
          messageEl.parentNode.removeChild(messageEl);
        }
      }, 300);
    }, 3000);
  }
}

function initScrollFadeIn() {
  const fadeElements = document.querySelectorAll('.scroll-fade-in');
  
  if (fadeElements.length === 0) {
    return;
  }
  
  const heroSection = document.querySelector('.ai-enhance-hero');
  
  fadeElements.forEach((element) => {
    if (heroSection && heroSection.contains(element)) {
      element.style.opacity = '1';
      element.style.transform = 'translateY(0)';
      element.classList.add('visible');
    } else {
      element.style.opacity = '0';
      element.style.transform = 'translateY(50px)';
      element.classList.remove('visible');
    }
  });
  
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
    const triggerPoint = windowHeight * 0.8; // Trigger when the element reaches 80% of the viewport height
    const fadeInRange = windowHeight * 0.6; // Fade-in range
    
    fadeElements.forEach((element) => {
      const rect = element.getBoundingClientRect();
      const elementTop = rect.top;
      const elementHeight = rect.height;
      const elementBottom = rect.bottom;
      
      const isPrivacyNotice = element.classList.contains('privacy-notice');

      if (isPrivacyNotice && element.classList.contains('visible')) {
        element.style.opacity = '1';
        element.style.transform = 'translateY(0)';
        return;
      }
      
      if (elementTop < triggerPoint && elementTop > -elementHeight) {
        let progress = Math.max(0, Math.min(1, (triggerPoint - elementTop) / fadeInRange));
        
        element.style.opacity = progress.toString();
        element.style.transform = `translateY(${50 * (1 - progress)}px)`;
        
        if (progress >= 0.9 || (isPrivacyNotice && progress >= 0.5)) {
          element.classList.add('visible');
        }
      } else if (elementTop <= -elementHeight) {
        element.style.opacity = '1';
        element.style.transform = 'translateY(0)';
        element.classList.add('visible');
      } else {
        element.style.opacity = '0';
        element.style.transform = 'translateY(50px)';
        element.classList.remove('visible');
      }
    });
  }
}


/**
 * Get a saved API key
 * @param {string} keyName - API key name ('openai')
 * @returns {string|null} - The API key string, or null if it does not exist.
 */
function getAPIKey(keyName) {
  if (!keyName || typeof keyName !== 'string') {
    console.warn('getAPIKey: invalid keyName parameter');
    return null;
  }
  
  if (keyName !== 'openai') {
    console.warn('getAPIKey: only "openai" is currently supported');
    return null;
  }
  
  const key = localStorage.getItem(`api_key_${keyName}`);
  return key || null;
}

/**
 * Check whether an API key exists
 * @param {string} keyName - API key name ('openai')
 * @returns {boolean} - True if the key exists, otherwise false.
 */
function hasAPIKey(keyName) {
  return getAPIKey(keyName) !== null;
}

/**
 * Get all saved API keys
 * @returns {Object} - An object containing all saved API keys.
 */
function getAllAPIKeys() {
  return {
    openai: getAPIKey('openai')
  };
}

window.getAPIKey = getAPIKey;
window.hasAPIKey = hasAPIKey;
window.getAllAPIKeys = getAllAPIKeys;
