
process fetch_dataset {
    publishDir params.outdir, mode: 'copy', saveAs: { file -> "${project_id}.${file}" }
    tag "${project_id}"

    input:
    val(project_id)
    val(token)

    output:
    tuple val(project_id), path('data.txt'), path('meta.json'), emit: datasets

    script:
    """
    fetch-dataset.py --project_id ${project_id} --token ${token}
    """
}
