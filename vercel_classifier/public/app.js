const form = document.getElementById('form');
const button = document.getElementById('submit');
const error = document.getElementById('error');
const result = document.getElementById('result');
const provider = document.getElementById('provider');
const model = document.getElementById('model');
const customModel = document.getElementById('custom-model');
const customWrap = document.getElementById('custom-model-wrap');
const status = document.getElementById('provider-status');
let configuration;
let busy = false;

function clearResult() {
  error.hidden = true;
  result.hidden = true;
}
function updateStatus() {
  const selected = configuration.providers[provider.value];
  status.textContent = selected.configured
    ? `${selected.label} key is configured on the server.`
    : `${selected.label} key is not configured. Choose another provider.`;
  button.disabled = busy || !selected.configured;
}
function updateModelInput() {
  customWrap.hidden = model.value !== 'custom';
  customModel.required = !customWrap.hidden;
  clearResult();
}
function updateProvider() {
  const selected = configuration.providers[provider.value];
  model.replaceChildren();
  for (const id of selected.models) model.add(new Option(id, id));
  model.add(new Option('Custom model ID…', 'custom'));
  model.value = selected.default_model;
  customModel.value = '';
  updateModelInput();
  updateStatus();
}
provider.addEventListener('change', updateProvider);
model.addEventListener('change', updateModelInput);
customModel.addEventListener('input', clearResult);

async function loadConfiguration() {
  try {
    const response = await fetch('/api/config', {cache: 'no-store', signal: AbortSignal.timeout(10000)});
    if (!response.ok) throw new Error('Could not load provider settings. Refresh the page to retry.');
    configuration = await response.json();
    provider.replaceChildren();
    for (const [id, value] of Object.entries(configuration.providers)) {
      provider.add(new Option(value.label, id));
    }
    provider.value = configuration.default_provider;
    provider.disabled = false;
    model.disabled = false;
    updateProvider();
  } catch (exc) {
    status.textContent = 'Provider settings are unavailable. Refresh the page to retry.';
    error.textContent = exc.message;
    error.hidden = false;
  }
}

form.addEventListener('submit', async event => {
  event.preventDefault();
  if (busy || !configuration) return;
  clearResult();
  busy = true;
  button.disabled = true;
  provider.disabled = true;
  model.disabled = true;
  customModel.disabled = true;
  button.textContent = 'Classifying…';
  try {
    const selectedModel = model.value === 'custom' ? customModel.value.trim() : model.value;
    const response = await fetch('/api/classify', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({description: document.getElementById('description').value,
                            provider: provider.value, model: selectedModel}),
      signal: AbortSignal.timeout(35000)
    });
    let data;
    try { data = await response.json(); }
    catch { throw new Error('The service could not respond. Please try again shortly.'); }
    if (!response.ok) throw new Error(data.error || 'Classification failed. Please try again.');
    document.getElementById('category').textContent = data.category;
    document.getElementById('attribution').textContent =
      `${configuration.providers[data.provider].label} · ${data.model}`;
    result.hidden = false;
  } catch (exc) {
    error.textContent = exc.name === 'TimeoutError'
      ? 'The request took too long. Please try again or choose another model.' : exc.message;
    error.hidden = false;
  } finally {
    busy = false;
    provider.disabled = false;
    model.disabled = false;
    customModel.disabled = false;
    button.textContent = 'Classify description';
    updateStatus();
  }
});
loadConfiguration();
