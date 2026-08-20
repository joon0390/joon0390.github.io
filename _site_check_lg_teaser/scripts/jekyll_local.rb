require "bundler/setup"
require "jekyll"
require "liquid/interrupts"

STANDARD_TAGS = [
  ["assign", "liquid/tags/assign", "Assign"],
  ["break", "liquid/tags/break", "Break"],
  ["capture", "liquid/tags/capture", "Capture"],
  ["case", "liquid/tags/case", "Case"],
  ["comment", "liquid/tags/comment", "Comment"],
  ["continue", "liquid/tags/continue", "Continue"],
  ["cycle", "liquid/tags/cycle", "Cycle"],
  ["decrement", "liquid/tags/decrement", "Decrement"],
  ["for", "liquid/tags/for", "For"],
  ["if", "liquid/tags/if", "If"],
  ["ifchanged", "liquid/tags/ifchanged", "Ifchanged"],
  ["increment", "liquid/tags/increment", "Increment"],
  ["raw", "liquid/tags/raw", "Raw"],
  ["tablerow", "liquid/tags/table_row", "TableRow"],
  ["unless", "liquid/tags/unless", "Unless"],
].freeze

STANDARD_TAGS.each do |tag_name, require_path, const_name|
  require require_path
  Liquid::Template.register_tag(tag_name, Liquid.const_get(const_name))
end

Liquid::Template.register_tag("include", Jekyll::Tags::IncludeTag)
Liquid::Template.register_tag("include_relative", Jekyll::Tags::IncludeRelativeTag)
Liquid::Template.register_tag("link", Jekyll::Tags::Link)
Liquid::Template.register_tag("post_url", Jekyll::Tags::PostUrl)
Liquid::Template.register_tag("highlight", Jekyll::Tags::HighlightBlock)

load Gem.bin_path("jekyll", "jekyll", Gem.loaded_specs.fetch("jekyll").version.to_s)
