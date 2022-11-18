# Contributing to cryosparc-tools

Thanks for taking the time to contribute! See the [Table of Contents](#table-of-contents) for different ways to help and details about how this project handles them. Please make sure to read the relevant section before making your contribution.

<!-- omit in toc -->

## Table of Contents

- [I Have a Question](#i-have-a-question)
- [I Want To Contribute](#i-want-to-contribute)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Enhancements](#suggesting-enhancements)
- [Your First Code Contribution](#your-first-code-contribution)
- [Improving The Documentation](#improving-the-documentation)
- [Styleguides](#styleguides)
- [Commit Messages](#commit-messages)

## I Have a Question

> If you want to ask a question, we assume that you have read the available [Documentation](https://tools.cryosparc.com).

Before asking a question, search for existing [Issues](https://github.com/cryoem-uoft/cryosparc-tools/issues) that might help. If additional clarification is needed for a relevant issue, write a question in this issue. Search the [CryoSPARC Discussion Forum](https://discuss.cryosparc.com) for additional help.

If you still have a question and need clarification, we recommend the following:

- Open an [Issue](https://github.com/cryoem-uoft/cryosparc-tools/issues/new).
- Provide as much context as you can about what you're running into.
- Provide project and platform versions (operating system, python version, pip/conda version, etc), depending on what seems relevant.

Structura will address the issue as soon as possible.

## I Want To Contribute

> ### Legal Notice <!-- omit in toc -->
>
> When contributing to this project, you must agree that you have authored 100% of the content, that you have the necessary rights to the content and that the content you contribute may be provided under the project license.

### Reporting Bugs

<!-- omit in toc -->

#### Before Submitting a Bug Report

Please complete the following steps in advance to help us fix any potential bug as fast as possible.

- Make sure that you are using the latest version.
- Determine if your bug is really a bug and not an error on your side e.g. using incompatible environment components/versions (Make sure that you have read the [documentation](https://tools.cryosparc.com). If you are looking for support, you might want to check [this section](#i-have-a-question)).
- To see if others have experienced (and potentially solved) the same issue you are having, check if there is not already a bug report existing for your bug or error in the [bug tracker](https://github.com/cryoem-uoft/cryosparc-toolsissues?q=label%3Abug).
- Also make sure to search the internet (including the [discussion forum](https://discuss.cryosparc.com)) to see if others outside of GitHub have discussed the issue.
- Collect information about the bug:
  - Stack trace (Traceback)
  - OS, Platform and Version (Windows, Linux, macOS, x86, ARM)
  - Version of the interpreter, runtime environment, package manager, depending on what seems relevant.
- Possibly your input and the output
- Can you reliably reproduce the issue? And can you also reproduce it with older versions?

<!-- omit in toc -->

#### How Do I Submit a Good Bug Report?

> Never report security related issues, vulnerabilities or bugs including sensitive information to the issue tracker, or elsewhere in public. Instead sensitive bugs must be sent by email to info@structura.bio.

<!-- You may add a PGP key to allow the messages to be sent encrypted as well. -->

We use GitHub issues to track bugs and errors. If you run into an issue with the project:

- Open an [Issue](https://github.com/cryoem-uoft/cryosparc-tools/issues/new).
- Explain the behavior you would expect and the actual behavior.
- Please provide as much context as possible and describe the _reproduction steps_ that someone else can follow to recreate the issue on their own. This usually includes your code. For good bug reports you should isolate the problem and create a reduced test case.
- Provide the information you collected in the previous section.

Once it's filed:

- The project team will label the issue accordingly.
- A team member will try to reproduce the issue with your provided steps. If there are no reproduction steps or no obvious way to reproduce the issue, the team will ask you for those steps. Bugs without clear reproduction steps will not be addressed until those steps are provided.
- If the team is able to reproduce the issue, it will be assigned to a release milestone, and the fix will be implemented in a future version.

<!-- You might want to create an issue template for bugs and errors that can be used as a guide and that defines the structure of the information to be included. If you do so, reference it here in the description. -->

### Suggesting Enhancements

This section guides you through submitting an enhancement suggestion for cryosparc-tools, **including completely new features and minor improvements to existing functionality**. Following these guidelines will help maintainers and the community to understand your suggestion and find related suggestions.

<!-- omit in toc -->

#### Before Submitting an Enhancement

- Make sure that you are using the latest version.
- Read the [documentation](https://tools.cryosparc.com) carefully and find out if the functionality is already covered, maybe by an individual configuration.
- Perform a [search](https://github.com/cryoem-uoft/cryosparc-tools/issues) to see if the enhancement has already been suggested. If it has, add a comment to the existing issue instead of opening a new one.
- Find out whether your idea fits with the scope and aims of the project. It's up to you to make a strong case to convince the project's developers of the merits of this feature. Keep in mind that we want features that will be useful to the majority of our users and not just a small subset. If you're just targeting a minority of use cases, consider writing an add-on/plugin library.

<!-- omit in toc -->

#### How Do I Submit a Good Enhancement Suggestion?

Enhancement suggestions are tracked as [GitHub issues](https://github.com/cryoem-uoft/cryosparc-tools/issues).

- Use a **clear and descriptive title** for the issue to identify the suggestion.
- Provide a **step-by-step description of the suggested enhancement** in as many details as possible.
- **Describe the current behavior** and **explain which behavior you expected to see instead** and why. At this point you can also tell which alternatives do not work for you.
- You may want to **include screenshots and animated GIFs** which help you demonstrate the steps or point out the part which the suggestion is related to.
- **Explain why this enhancement would be useful** to most cryosparc-tools users. You may also want to point out the other projects that solved it better and which could serve as inspiration.

<!-- You might want to create an issue template for enhancement suggestions that can be used as a guide and that defines the structure of the information to be included. If you do so, reference it here in the description. -->

### Your First Code Contribution

This repository uses Git version control and is hosted on GitHub,

[Fork](https://github.com/cryoem-uoft/cryosparc-tools/fork) this repository to your GitHub account and clone to your development machine. Follow the steps in the [Development section in README.md](README.md#development) to set up your development environment.

Make your changes to the `main` branch. Test your changes by running
the built-in pytest tests:

```
pytest
```

Include more tests in your change set if appropriate.

Run the following commands to check the code for issues related to type-safety, linting and formatting.

```sh
pyright
ruff .
black .
```

Commit your changes and [submit a pull request](https://github.com/cryoem-uoft/cryosparc-tools/compare). The pull request will be reviewed by someone on the Stuctura team. Check your GitHub notifications periodically for change requests. Once the pull request is approved, it will be merged to the `main` branch and deployed in the next release.

### Improving The Documentation

Documentation is located in the `docs` directory and is powered by [Jupyter Book](https://jupyterbook.org/en/stable/intro.html).

Inline source documentation is compiled to HTML via [Sphinx](https://www.sphinx-doc.org/en/master/index.html) and uses [Google Style Python docstrings](https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html#example-google).

## Styleguides

cryosparc-tools uses [black](https://black.readthedocs.io/en/stable/) for
formatting source code. Configuration options for black are included in
pyproject.toml

To format code with the required style, run `black .` in the cryosparc-tools
directory.

### Commit Messages

The first line of a Git commit messages must use one of the following patterns:

```
feat: The new feature you're adding to a particular application
fix: A bug fix
style: Feature and updates related to styling
refactor: Refactoring a specific section of the codebase
test: Everything related to testing
docs: Everything related to documentation
chore: Regular code maintenance.
```

## Writing tools, libraries and addons

cryosparc-tools is a Python package and may be used build custom cryo-EM
workflows, write results from an external tool back into CryoSPARC and extend
CryoSPARC functionality.

If you publish an open-source tool that uses this package to GitHub, add the `cryosparc-tools` topic to your repository so others may discover it. [Browse tagged packages here](https://github.com/topics/cryosparc-tools).

<!-- omit in toc -->

## Attribution

This guide is based on the **contributing-gen**. [Make your own](https://generator.contributing.md)!
