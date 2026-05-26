# Work around local Windows/Ruby environments where Liquid's built-in tags
# are not eagerly registered before Jekyll starts rendering layouts.
begin
  liquid_spec = Gem.loaded_specs["liquid"] || Gem::Specification.find_by_name("liquid")
  tag_glob = File.join(liquid_spec.full_gem_path, "lib", "liquid", "tags", "*.rb")

  Dir.glob(tag_glob).sort.each do |tag_file|
    require tag_file
  end
rescue Gem::LoadError
  warn "Liquid gem could not be located while bootstrapping built-in tags."
end
