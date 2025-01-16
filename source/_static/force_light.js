document.addEventListener('DOMContentLoaded', function() {
    // Force light mode
    document.documentElement.dataset.theme = 'light';
    localStorage.setItem('theme', 'light');
    
    // Override the theme detection
    window.matchMedia('(prefers-color-scheme: dark)').matches = false;
    
    // Remove any dark mode classes
    document.documentElement.classList.remove('dark');
    
    // Prevent theme changes
    const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.attributeName === 'data-theme') {
                document.documentElement.dataset.theme = 'light';
            }
        });
    });

    observer.observe(document.documentElement, {
        attributes: true
    });
});