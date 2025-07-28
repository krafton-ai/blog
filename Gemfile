source 'https://rubygems.org'
ruby '~> 3.2.0'
group :jekyll_plugins do
    gem 'classifier-reborn'
    gem 'jekyll'
    gem 'jekyll-archives'
    gem 'jekyll-feed'
    gem 'jekyll-paginate-v2'
    gem 'jekyll-scholar'
    gem 'jekyll-sitemap'
    gem 'jekyll-target-blank'
    gem 'webrick'
end
group :other_plugins do
    gem 'css_parser'
    gem 'feedjira'
    gem 'httparty'
    # Only include `uri` gem when running in CI (e.g., GitHub Actions)
    gem 'uri', '0.10.1' if ENV['CI']
end