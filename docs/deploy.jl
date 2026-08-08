using Documenter

struct CodebergDeploy <: Documenter.DeployConfig
    https_url::Any
end
Documenter.deploy_folder(
    ::CodebergDeploy;
    repo,
    branch,
    tag_prefix,
    kwargs...,
) = Documenter.DeployDecision(;
    all_ok = true,
    branch,
    repo,
    subfolder = tag_prefix,
)
Documenter.authentication_method(::CodebergDeploy) = Documenter.HTTPS
Documenter.authenticated_repo_url(cd::CodebergDeploy) = cd.https_url

deploydocs(;
    repo = "codeberg.org/a5s/PointCloudRegistration.jl.git",
    branch = "pages",
    deploy_config = CodebergDeploy(
        "https://codeberg.org/a5s/PointCloudRegistration.jl.git",
    ),
    # devurl = "main",
)
