document.addEventListener('DOMContentLoaded', () => {
  const modules = [
    'home', 'lexer', 'semantic', 'symtab', 'types', 'error',
    'parser', 'nodes_ast', 'interpreter', 'ast_to_json',
    'reader_script', 'main'
  ];

  const sections = document.querySelectorAll('.page-section');
  const navLinks = document.querySelectorAll('.nav-link');

  function activate(id) {
    sections.forEach(section => {
      section.classList.remove('active');
    });
    const activeSection = document.getElementById(id);
    if (activeSection) {
      activeSection.classList.add('active');
    } else {
      console.error(`Section with ID '${id}' not found to activate.`);
    }

    navLinks.forEach(navLink => {
      const targetId = 'page-' + navLink.getAttribute('href').substring(1);
      if (targetId === id) {
        navLink.classList.add('active');
      } else {
        navLink.classList.remove('active');
      }
    });
  }

  modules.forEach(id => {
    fetch(`Markdown/${id}.md`)
      .then(response => {
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status} for Markdown/${id}.md`);
        }
        return response.text();
      })
      .then(markdown => {
        const html = marked.parse(markdown);
        const section = document.getElementById(`page-${id}`);
        if (section) {
          section.innerHTML = `
            <section class="bg-white p-6 rounded-lg shadow-md mb-6">
              <div class="md mb-4">
                ${html}
              </div>
            </section>
          `;
          section.querySelectorAll('pre code').forEach((block) => {
            hljs.highlightElement(block);
          });
        } else {
          console.error(`Element with ID 'page-${id}' not found.`);
        }
      })
      .catch(error => console.error(`Error loading or processing markdown for ${id}:`, error));
  });

  const initialHash = window.location.hash;
  let initialTarget = 'home';
  if (initialHash) {
    const hashTarget = initialHash.substring(1);
    if (modules.includes(hashTarget)) {
      initialTarget = hashTarget;
    }
  }
  setTimeout(() => activate('page-' + initialTarget), 100);

  navLinks.forEach(link => {
    link.addEventListener('click', e => {
      e.preventDefault();
      const target = link.getAttribute('href').substring(1);
      activate('page-' + target);
      // Descomenta la siguiente línea para actualizar la URL sin recargar la página
      history.pushState(null, null, `#${target}`);
    });
  });

  // Añadir manejo del evento popstate para los botones de atrás/adelante del navegador
  window.addEventListener('popstate', () => {
    const hash = window.location.hash;
    let target = 'home'; // Por defecto a home si no hay hash
    if (hash) {
        const hashTarget = hash.substring(1);
        if (modules.includes(hashTarget)) {
            target = hashTarget;
        }
    }
    activate('page-' + target);
  });

}); // Fin de DOMContentLoaded
