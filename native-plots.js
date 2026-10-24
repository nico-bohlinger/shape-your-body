const plotNamespace = 'http://www.w3.org/2000/svg';

function drawPlotNode([tag, attributes, children]) {
    const element = document.createElementNS(plotNamespace, tag);
    for (const [name, value] of Object.entries(attributes)) element.setAttribute(name, value);
    for (const child of children) element.append(drawPlotNode(child));
    return element;
}

function applyPlotColors(svg, palette) {
    palette.forEach((colors, index) => svg.style.setProperty(`--plot-color-${index}`, colors[1]));
}

async function loadPlot(figure) {
    const name = figure.dataset.plot;
    const surface = figure.querySelector('.plot-surface');
    try {
        const response = await fetch(`plots/${name}.json?v=light-v1`);
        if (!response.ok) throw new Error(`Unable to load ${name}`);
        const chart = await response.json();
        const svg = drawPlotNode(chart.drawing);
        svg.setAttribute('aria-hidden', 'true');
        svg.setAttribute('focusable', 'false');
        applyPlotColors(svg, chart.palette);
        surface.replaceChildren(svg);
        figure.dataset.rendered = 'svg';
    } catch (error) {
        const image = document.createElement('img');
        image.alt = '';
        image.src = `images/${name}.svg?v=light-v1`;
        surface.replaceChildren(image);
        figure.dataset.rendered = 'fallback';
        console.warn(error);
    }
}

const plotObserver = new IntersectionObserver(entries => {
    for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        plotObserver.unobserve(entry.target);
        loadPlot(entry.target);
    }
}, { rootMargin: '400px' });
for (const figure of document.querySelectorAll('.native-plot')) plotObserver.observe(figure);
