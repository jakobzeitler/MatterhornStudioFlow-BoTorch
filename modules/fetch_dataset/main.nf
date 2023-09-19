
process fetch_dataset {
    publishDir params.outdir, mode: 'copy', saveAs: { file -> "${project_id}.${file}" }
    tag "${project_id}"

    input:
    val(project_id)
    val(opt_run_id)
    val(token)
    val(base_url)

    output:
    tuple val(project_id), path('data.txt'), path('meta.json'), emit: datasets

    script:
    """
    fetch-dataset.py --project_id ${project_id} --opt_run_id ${opt_run_id} --token ${token} --base_url ${base_url}
    """
}
