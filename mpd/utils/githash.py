import git


def get_git_hash_short():
    repo = git.Repo(search_parent_directories=True)
    githash = repo.git.rev_parse(repo.head, short=True)
    return githash
