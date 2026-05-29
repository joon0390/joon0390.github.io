# Work around local Windows/Ruby environments where Liquid's built-in tags
# are not eagerly registered before Jekyll starts rendering layouts.
begin
  liquid_spec = Gem.loaded_specs["liquid"] || Gem::Specification.find_by_name("liquid")
  tags_dir = File.join(liquid_spec.full_gem_path, "lib", "liquid", "tags")

  Dir.children(tags_dir).grep(/\.rb\z/).sort.reject { |tag_file| tag_file == "include.rb" }.each do |tag_file|
    require File.join(tags_dir, tag_file)
  end
rescue Gem::LoadError
  warn "Liquid gem could not be located while bootstrapping built-in tags."
end
