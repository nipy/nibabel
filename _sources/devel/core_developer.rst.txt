.. _core_dev:

Core Developer Guide
====================

As a core developer, you should continue making pull requests
in accordance with the :ref:`chap_devguide`.
You are responsible for shepherding other contributors through the review process.
You also have the ability to merge or approve other contributors' pull requests.

Reviewing
---------

How to Conduct A Good Review
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Always* be kind to contributors. Nearly all of Nibabel is
volunteer work, for which we are tremendously grateful. Provide
constructive criticism on ideas and implementations, and remind
yourself of how it felt when your own work was being evaluated as a
novice.

Nibabel strongly values mentorship in code review.  New users
often need more handholding, having little to no git
experience. Repeat yourself liberally, and, if you don’t recognize a
contributor, point them to our development guide, or other GitHub
workflow tutorials around the web. Do not assume that they know how
GitHub works (e.g., many don't realize that adding a commit
automatically updates a pull request). Gentle, polite, kind
encouragement can make the difference between a new core developer and
an abandoned pull request.

When reviewing, focus on the following:

1. **API:** The API is what users see when they first use
   Nibabel. APIs are difficult to change once released, so
   should be simple, consistent with other parts of the library, and
   should avoid side-effects such as changing global state or modifying
   input variables.

2. **Documentation:** Any new feature should have a tutorial
   example that not only illustrates but explains it.

3. **The algorithm:** You should understand the code being modified or
   added before approving it.  (See `Merge Only Changes You
   Understand`_ below.) Implementations should do what they claim,
   and be simple, readable, and efficient.

4. **Tests:** All contributions to the library *must* be tested, and
   each added line of code should be covered by at least one test. Good
   tests not only execute the code, but explore corner cases.  It is tempting
   not to review tests, but please do so.

Other changes may be *nitpicky*: spelling mistakes, formatting,
etc. Do not ask contributors to make these changes, and instead
make the changes by `pushing to their branch
<https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/committing-changes-to-a-pull-request-branch-created-from-a-fork>`__,
or using GitHub’s `suggestion
<https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/commenting-on-a-pull-request>`__
`feature
<https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/incorporating-feedback-in-your-pull-request>`__.
(The latter is preferred because it gives the contributor a choice in
whether to accept the changes.)

Please add a note to a pull request after you push new changes; GitHub
may not send out notifications for these.

Merge Only Changes You Understand
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Long-term maintainability* is an important concern.  Code doesn't
merely have to *work*, but should be *understood* by multiple core
developers.  Changes will have to be made in the future, and the
original contributor may have moved on.

Therefore, *do not merge a code change unless you understand it*. Ask
for help freely: we have a long history of consulting community
members, or even external developers, for added insight where needed,
and see this as a great learning opportunity.

While we collectively "own" any patches (and bugs!) that become part
of the code base, you are vouching for changes you merge.  Please take
that responsibility seriously.

Closing issues and pull requests
--------------------------------

Sometimes, an issue must be closed that was not fully resolved. This can be
for a number of reasons:

- the person behind the original post has not responded to calls for
  clarification, and none of the core developers have been able to reproduce
  their issue;
- fixing the issue is difficult, and it is deemed too niche a use case to
  devote sustained effort or prioritize over other issues; or
- the use case or feature request is something that core developers feel
  does not belong in Nibabel,

among others. Similarly, pull requests sometimes need to be closed without
merging, because:

- the pull request implements a niche feature that we consider not worth the
  added maintenance burden;
- the pull request implements a useful feature, but requires significant
  effort to bring up to Nibabel's standards, and the original
  contributor has moved on, and no other developer can be found to make the
  necessary changes; or
- the pull request makes changes that make maintenance harder, such as
  increasing the code complexity of a function significantly to implement a
  marginal speedup,

among others.

All these may be valid reasons for closing, but we must be wary not to alienate
contributors by closing an issue or pull request without an explanation. When
closing, your message should:

- explain clearly how the decision was made to close. This is particularly
  important when the decision was made in a community meeting, which does not
  have as visible a record as the comments thread on the issue itself;
- thank the contributor(s) for their work; and
- provide a clear path for the contributor or anyone else to appeal the
  decision.

These points help ensure that all contributors feel welcome and empowered to
keep contributing, regardless of the outcome of past contributions.

Further resources
-----------------

As a core member, you should be familiar with community and developer
resources such as:

-  Our :ref:`chap_devguide`
-  Our :ref:`community_guidelines`
-  `PEP8 <https://www.python.org/dev/peps/pep-0008/>`__ for Python style
-  `PEP257 <https://www.python.org/dev/peps/pep-0257/>`__ and the `NumPy
   documentation
   guide <https://numpy.org/doc/stable/docs/howto_document.html>`__
   for docstrings. (NumPy docstrings are a superset of PEP257. You
   should read both.)
-  The Nibabel `tag on NeuroStars <https://neurostars.org/tag/nibabel>`__
-  Our `mailing
   list <https://mail.python.org/mailman/listinfo/neuroimaging>`__

Please do monitor any of the resources above that you find helpful.

Acknowledgments
===============

This document is based on the `NetworkX Core Developer guide
<https://networkx.org/documentation/latest/developer/core_developer.html>`_.
