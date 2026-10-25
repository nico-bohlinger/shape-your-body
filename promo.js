const hero = document.getElementById('hero-film');
const toggle = document.getElementById('hero-toggle');
const motion = window.matchMedia('(prefers-reduced-motion: reduce)');
let requestedPlayback = !motion.matches && !navigator.connection?.saveData;
let heroVisible = false;

function updatePlayback() {
    if (requestedPlayback && heroVisible && !document.hidden) {
        const sources = [...hero.querySelectorAll('source')];
        if (sources.some(source => !source.hasAttribute('src'))) {
            for (const source of sources) source.src = source.dataset.src;
            hero.load();
        }
        hero.play().catch(() => {
            requestedPlayback = false;
            toggle.dataset.playing = 'false';
            toggle.setAttribute('aria-label', 'Play animation');
        });
    } else {
        hero.pause();
    }
}

new IntersectionObserver(([entry]) => {
    heroVisible = entry.isIntersecting;
    updatePlayback();
}, { threshold: 0.2 }).observe(hero);

for (const event of ['play', 'pause']) {
    hero.addEventListener(event, () => {
        toggle.dataset.playing = String(!hero.paused);
        toggle.setAttribute('aria-label', hero.paused ? 'Play animation' : 'Pause animation');
    });
}

toggle.addEventListener('click', () => {
    requestedPlayback = hero.paused;
    updatePlayback();
});

motion.addEventListener('change', () => {
    requestedPlayback = !motion.matches && !navigator.connection?.saveData;
    updatePlayback();
});

document.addEventListener('visibilitychange', updatePlayback);

const demo = document.getElementById('syb-demo');

const demoObserver = new IntersectionObserver(entries => {
    if (entries.some(entry => entry.isIntersecting)) {
        demoObserver.disconnect();
        import('./demo/demo.js?v=light-v3').then(module => module.selectDesign(null)).catch(() => {
            document.getElementById('syb-status').textContent = 'The interactive demo could not load. Reload to try again.';
        });
    }
}, { rootMargin: '100px' });
demoObserver.observe(demo);

const copyCitation = document.getElementById('copy-citation');
let citationCopyTimer;
copyCitation.addEventListener('click', async () => {
    clearTimeout(citationCopyTimer);
    const code = document.querySelector('.citation-container code');
    const label = copyCitation.querySelector('span');
    try {
        await navigator.clipboard.writeText(code.textContent.trim());
        label.textContent = 'Copied';
        copyCitation.classList.add('copied');
    } catch {
        const range = document.createRange();
        range.selectNodeContents(code);
        const selection = window.getSelection();
        selection.removeAllRanges();
        selection.addRange(range);
        label.textContent = 'Copy manually';
        copyCitation.classList.remove('copied');
    }
    citationCopyTimer = setTimeout(() => {
        label.textContent = 'Copy';
        copyCitation.classList.remove('copied');
    }, 2500);
});
