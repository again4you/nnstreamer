repo_sources = [
  'gstdatarepo.c',
  'gstdatareposrc.c',
  'gstdatareposink.c'
]

gstdatarepo_shared = shared_library('gstdatarepo',
  repo_sources,
  dependencies: [nnstreamer_dep, json_glib_dep],
  install: true,
  install_dir: plugins_install_dir
)

gstdatarepo_static = static_library('gstdatarepo',
  repo_sources,
  dependencies: [nnstreamer_dep, json_glib_dep],
  install: true,
  install_dir: nnstreamer_libdir
)

gstdatarepo_lib = gstdatarepo_shared
if get_option('default_library') == 'static'
  gstdatarepo_lib = gstdatarepo_static
endif

repo_dep = declare_dependency(link_with: gstdatarepo_lib)
