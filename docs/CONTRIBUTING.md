# Contributing to Video Analyzer

Thank you for your interest in contributing to the Video Analyzer project! This guide outlines the process for making contributions through pull requests.

## Before You Start

1. Read [docs/DESIGN.md](DESIGN.md) thoroughly to understand:
   - The project's architecture
   - Core components and their interactions
   - Design decisions and rationale
   - Implementation details

2. Familiarize yourself with the codebase:
   - Review the project structure
   - Understand the different modules
   - Check existing features and implementations

## Proposing Changes

1. Before creating a PR, start a discussion in the [GitHub Discussions](https://github.com/byjlw/video-analyzer/discussions) section:
   - Outline your proposed changes
   - Explain the motivation behind the changes
   - Describe your planned implementation approach
   - Wait for community feedback and maintainer input

2. Use the appropriate discussion category:
   - "Ideas" for new features
   - "Q&A" for questions about implementation
   - "Show and Tell" for sharing prototypes

## Making Changes

Once your proposal has been discussed and approved:

1. Fork the repository and create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Follow the project's coding standards:
   - Maintain consistent code style
   - Add appropriate documentation
   - Include type hints
   - Write clear commit messages

3. Add tests for new functionality:
   - Unit tests for individual components
   - Integration tests for feature workflows
   - Ensure all tests pass

4. Update documentation:
   - Add/update docstrings
   - Update relevant .md files
   - Add examples if applicable

## Submitting Pull Requests

1. Before submitting:
   - Ensure all tests pass
   - Update documentation
   - Add your changes to CHANGELOG.md
   - Rebase on latest main branch

2. Create a pull request:
   - Reference the discussion thread
   - Provide a clear description of changes
   - List any breaking changes
   - Include testing steps

3. PR description should include:
   - Link to the discussion thread
   - Summary of changes
   - Testing performed
   - Screenshots/videos if UI changes
   - Breaking changes (if any)

## Review Process

1. Maintainers will review your PR:
   - Code quality
   - Test coverage
   - Documentation
   - Design consistency

2. Address review feedback:
   - Make requested changes
   - Respond to comments
   - Update tests if needed

3. Once approved:
   - Squash commits if requested
   - Ensure branch is up to date
   - Wait for merge by maintainers

## Additional Guidelines

### Code Style
- Follow PEP 8 guidelines
- Use meaningful variable names
- Keep functions focused and concise
- Add type hints to function parameters
- Document complex logic

### Testing
- Write tests for new features
- Update existing tests if needed
- Ensure test coverage
- Test edge cases

### Documentation
- Keep documentation up to date
- Use clear, concise language
- Include code examples
- Update README.md if needed

### Commit Messages
- Use clear, descriptive messages
- Reference issues/discussions
- Follow conventional commits format:
  ```
  feat: add new feature X
  fix: resolve issue with Y
  docs: update contributing guidelines
  test: add tests for feature Z
  ```

## Getting Help

If you need help:
1. Check existing [discussions](https://github.com/byjlw/video-analyzer/discussions)
2. Start a new discussion if needed
3. Tag maintainers for urgent issues

## License

By contributing, you agree that your contributions will be licensed under the project's apache License.
