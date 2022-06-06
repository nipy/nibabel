.. _governance:

==============================
Governance and Decision Making
==============================

Abstract
========

Nibabel is a consensus-based community project. Anyone with an interest in the
project can join the community, contribute to the project design, and
participate in the decision making process. This document describes how that
participation takes place, how to find consensus, and how deadlocks are
resolved.

Roles And Responsibilities
==========================

The Community
-------------

The Nibabel community consists of anyone using or working with the project
in any way.

Contributors
------------

Any community member can become a contributor by interacting directly with the
project in concrete ways, such as:

- proposing a change to the code or documentation via a GitHub pull request;
- reporting issues on our
  `GitHub issues page <https://github.com/nipy/nibabel/issues>`_;
- discussing the design of the library, website, or tutorials on the
  `mailing list <https://mail.python.org/mailman/listinfo/neuroimaging>`_,
  or in existing issues and pull requests; or
- reviewing
  `open pull requests <https://github.com/nipy/nibabel/pulls>`_,

among other possibilities. By contributing to the project, community members
can directly help to shape its future.

Contributors should read the :ref:`chap_devguide` and our :ref:`community_guidelines`.

Core Developers
---------------

Core developers are community members that have demonstrated continued
commitment to the project through ongoing contributions. They
have shown they can be trusted to maintain Nibabel with care. Becoming a
core developer allows contributors to merge approved pull requests, cast votes
for and against merging a pull request, and be involved in deciding major
changes to the API, and thereby more easily carry on with their project related
activities.

Core developers:

================    =============
Name                GitHub user
================    =============
Chris Markiewicz    effigies
Matthew Brett       matthew-brett
Oscar Esteban       oesteban
================    =============

Core developers also appear as team members on the `Nibabel Core Team page
<https://github.com/orgs/nipy/teams/nibabel-core-developers/members>`_ and can
be messaged ``@nipy/nibabel-core-developers``. We expect core developers to
review code contributions while adhering to the :ref:`core_dev`.

New core developers can be nominated by any existing core developer. Discussion
about new core developer nominations is one of the few activities that takes
place on the project's private management list. The decision to invite a new
core developer must be made by “lazy consensus”, meaning unanimous agreement by
all responding existing core developers. Invitation must take place at least
one week after initial nomination, to allow existing members time to voice any
objections.

.. _steering_council:

Steering Council
----------------

The Steering Council (SC) members are current or former core developers who
have additional responsibilities to ensure the smooth running of the project.
SC members are expected to participate in strategic planning, approve changes
to the governance model, and make decisions about funding granted to the
project itself. (Funding to community members is theirs to pursue and manage.)
The purpose of the SC is to ensure smooth progress from the big-picture
perspective. Changes that impact the full project require analysis informed by
long experience with both the project and the larger ecosystem. When the core
developer community (including the SC members) fails to reach such a consensus
in a reasonable timeframe, the SC is the entity that resolves the issue.

The steering council is:

=================== =============
Name                GitHub user
=================== =============
Chris Markiewicz    effigies
Matthew Brett       matthew-brett
Michael Hanke       mih
Yaroslav Halchenko  yarikoptic
=================== =============

Steering Council members also appear as team members on the `Nibabel Steering
Council Team page
<https://github.com/orgs/nipy/teams/nibabel-steering-council/members>`_ and
can be messaged ``@nipy/nibabel-steering-council``.

Decision Making Process
=======================

Decisions about the future of the project are made through discussion with all
members of the community. All non-sensitive project management discussion takes
place on the project
`mailing list <https://mail.python.org/mailman/listinfo/neuroimaging>`_
and the `issue tracker <https://github.com/nipy/nibabel/issues>`_.
Occasionally, sensitive discussion may occur on a private list.

Decisions should be made in accordance with our :ref:`mission_and_values`.

Nibabel uses a *consensus seeking* process for making decisions. The group
tries to find a resolution that has no open objections among core developers.
Core developers are expected to distinguish between fundamental objections to a
proposal and minor perceived flaws that they can live with, and not hold up the
decision making process for the latter.  If no option can be found without
an objection, the decision is escalated to the SC, which will itself use
consensus seeking to come to a resolution. In the unlikely event that there is
still a deadlock, the proposal will move forward if it has the support of a
simple majority of the SC. Any proposal must be described by a Nibabel :ref:`biap`.

Decisions (in addition to adding core developers and SC membership as above)
are made according to the following rules:

- **Minor documentation changes**, such as typo fixes, or addition / correction
  of a sentence (but no change of the Nibabel landing page or the “about”
  page), require approval by a core developer *and* no disagreement or
  requested changes by a core developer on the issue or pull request page (lazy
  consensus). We expect core developers to give “reasonable time” to others to
  give their opinion on the pull request if they’re not confident others would
  agree.

- **Code changes and major documentation changes** require agreement by *one*
  core developer *and* no disagreement or requested changes by a core developer
  on the issue or pull-request page (lazy consensus).

- **Changes to the API principles** require a :ref:`biap` and follow the
  decision-making process outlined above.

- **Changes to this governance model or our mission and values** require
  a :ref:`biap` and follow the decision-making process outlined above, *unless*
  there is unanimous agreement from core developers on the change.

If an objection is raised on a lazy consensus, the proposer can appeal to the
community and core developers and the change can be approved or rejected by
escalating to the SC, and if necessary, a BIAP (see below).

.. _biap:

Enhancement Proposals (BIAPs)
=============================

Any proposals for enhancements of Nibabel should be written as a formal BIAP
following the template :ref:`biap_template`. The BIAP must be made public and
discussed before any vote is taken. The discussion must be summarized by a key
advocate of the proposal in the appropriate section of the BIAP. Once this
summary is made public and after sufficient time to allow the core team to
understand it, they vote.

The workflow of a BIAP is detailed in :ref:`biap0`.

A list of all existing BIAPs is available :ref:`here <biap_list>`.

Acknowledgments
===============

Many thanks to Jarrod Millman, Dan Schult and the Scikit-Image team for the
`draft on which we based this document
<https://networkx.github.io/documentation/latest/developer/nxeps/nxep-0001.html>`_.
