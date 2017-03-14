rm -rf *.html _sources _modules _static
git checkout master docs/build/html
mv docs/build/html/* .
rm -r docs
git add .
git commit -m "updated html"
git push origin gh-pages
