#!/usr/bin/env nextflow 

/*
 * Copyright (c) 2022, Seqera Labs.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * This Source Code Form is "Incompatible With Secondary Licenses", as
 * defined by the Mozilla Public License, v. 2.0.
 *
 */
import groovy.json.JsonSlurper

include { fetch_dataset } from './modules/fetch_dataset'

log.info """
    M L - H Y P E R O P T   P I P E L I N E
    =======================================
    fetch_dataset   : ${params.fetch_dataset}
    base_url  : ${params.base_url}
    token           : ${params.token}
    project_id    : ${params.project_id}
    opt_run_id    : ${params.opt_run_id}

    outdir          : ${params.outdir}
    """


/* 
 * main script flow
 */
workflow {
    // fetch dataset if specified
    if ( params.fetch_dataset == true ) {
        dataset = fetch_dataset(params.project_id, params.opt_run_id, params.token, params.base_url)

    }

}


/* 
 * completion handler
 */
workflow.onComplete {
	log.info ( workflow.success ? '\nDone!' : '\nOops .. something went wrong' )
}
