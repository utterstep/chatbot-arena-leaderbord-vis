// Chart instance
let eloChart = null;

// Color palette for different categories
const colors = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
];

// Function to load and process data
async function loadData() {
    try {
        const response = await fetch('elo_data.json');
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Error loading data:', error);
        return [];
    }
}

// Function to populate category dropdown
function populateCategories(data) {
    const categories = [...new Set(data.map(item => item.category))].sort();
    const select = document.getElementById('category-select');

    // Clear existing options
    select.innerHTML = '';

    categories.forEach(category => {
        const option = document.createElement('option');
        option.value = category;
        option.text = category;
        select.appendChild(option);
    });

    // Set default selections
    select.value = ['text_full', 'text_coding'];
}

// Function to destroy existing chart
function destroyChart() {
    if (eloChart) {
        eloChart.destroy();
        eloChart = null;
    }
}

// Function to update the chart
function updateChart(data, selectedCategories) {
    // Destroy existing chart
    destroyChart();

    const ctx = document.getElementById('eloChart').getContext('2d');

    // Filter data for selected categories
    const filteredData = data.filter(item => selectedCategories.includes(item.category));

    // Prepare datasets for Chart.js
    const datasets = selectedCategories.map((category, index) => {
        const categoryData = filteredData.filter(item => item.category === category);
        return {
            label: category,
            data: categoryData.map(item => ({
                x: new Date(item.date),
                y: item.max_elo,
                model: item.max_model
            })),
            borderColor: colors[index % colors.length],
            backgroundColor: colors[index % colors.length],
            pointRadius: 4,
            pointHoverRadius: 6,
            tension: 0.1
        };
    });

    // Create new chart
    eloChart = new Chart(ctx, {
        type: 'line',
        data: { datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Maximum Elo Rating Over Time'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const dataPoint = context.raw;
                            return [
                                `Category: ${context.dataset.label}`,
                                `Elo Rating: ${dataPoint.y.toFixed(2)}`,
                                `Model: ${dataPoint.model}`
                            ];
                        }
                    }
                }
            },
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'day',
                        displayFormats: {
                            day: 'yyyy-MM-dd'
                        }
                    },
                    title: {
                        display: true,
                        text: 'Date'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Elo Rating'
                    }
                }
            }
        }
    });
}

// Function to update top models information
function updateTopModels(data, selectedCategories) {
    const topModelsContainer = document.getElementById('top-models-info');
    topModelsContainer.innerHTML = '';

    selectedCategories.forEach(category => {
        const categoryData = data.filter(item => item.category === category);
        if (categoryData.length > 0) {
            // Get the latest data point
            const latestData = categoryData.reduce((latest, current) =>
                new Date(current.date) > new Date(latest.date) ? current : latest
            );

            const modelCard = document.createElement('div');
            modelCard.className = 'model-card';
            modelCard.innerHTML = `
                <h4>${category}</h4>
                <p>Top model: ${latestData.max_model}</p>
                <p>Elo rating: ${latestData.max_elo.toFixed(2)}</p>
                <p>Date: ${new Date(latestData.date).toLocaleDateString()}</p>
            `;
            topModelsContainer.appendChild(modelCard);
        }
    });
}

// Initialize the application
async function init() {
    const data = await loadData();
    if (data.length === 0) {
        console.error('No data available');
        return;
    }

    populateCategories(data);

    // Set up event listener for category selection
    const categorySelect = document.getElementById('category-select');
    categorySelect.addEventListener('change', () => {
        const selectedCategories = Array.from(categorySelect.selectedOptions).map(option => option.value);
        updateChart(data, selectedCategories);
        updateTopModels(data, selectedCategories);
    });

    // Initial chart render
    const initialCategories = ['text_full', 'text_coding'];
    updateChart(data, initialCategories);
    updateTopModels(data, initialCategories);
}

// Start the application
document.addEventListener('DOMContentLoaded', init);
