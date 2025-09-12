using Documenter

pseudo_remote = abspath(".pseudo_remote")
if !isdir(pseudo_remote)
    mkdir(pseudo_remote)
    cd(pseudo_remote) do 
        run(`$(Documenter.git()) init --bare`)
    end
end

struct LocalDeploy <: Documenter.DeployConfig end
Documenter.deploy_folder(::LocalDeploy; repo, branch, kwargs...) = Documenter.DeployDecision(;
    all_ok = true,
    branch,
    repo,
    subfolder = "main",
)
Documenter.authentication_method(::LocalDeploy) = Documenter.HTTPS
Documenter.authenticated_repo_url(::LocalDeploy) = pseudo_remote

deploydocs(
    repo = pseudo_remote,
    branch = "main",
    deploy_config = LocalDeploy(),
    devurl = "main",
)

deploy_dir = "public"
if isdir(deploy_dir)
    cd(deploy_dir) do
        run(`$(Documenter.git()) pull`)
    end
else
    run(`$(Documenter.git()) clone $pseudo_remote $deploy_dir`)
end

items = filter(!endswith(".git"), readdir("public/"; join = true))
run(`scp -r $items andreask@perseus.uberspace.de:html/PointCloudRegistration.jl`)
