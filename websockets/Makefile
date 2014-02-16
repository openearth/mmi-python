# create bundle.js by install ndarray using npm and then browerifying it
js/bundle.js:
	npm install ndarray ndarray-fill zeros ndarray-pack ndarray-unpack ndarray-ops browerify
	browserify -r ndarray -r ndarray-fill -r zeros -r ndarray-pack -r ndarray-unpack -r ndarray-ops > js/bundle.js
