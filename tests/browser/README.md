# Browser-flow tests

`app.js` is the labelling page's script, copied verbatim from the built page.
`harness.js` stubs just enough DOM to run it under node; `flow.js` drives a
session and asserts on the result.

    node tests/browser/harness.js tests/browser/app.js

Why this exists: the first build of point-marking shipped a page where circles
could be placed but pressing Yes neither asked for them nor acknowledged them.
Every static check passed -- valid HTML, valid JS, right ids, right columns --
because the defect was in the *flow*, which nothing static can see. These tests
press the keys.
